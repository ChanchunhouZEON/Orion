// This code is part of the Parlay Project
// Copyright (c) 2024 Guy Blelloch, Magdalen Dobson and the Parlay team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//#define Flexible
#pragma once
#ifdef Flexible
#include "simpleGraph.h"
#else

#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"

#include "types.h"
#include "large_io.h"

namespace parlayANN {
  
template<typename indexType>
struct edgeRange{

  size_t size() const {return edges[0];}

  indexType id() const {return id_;}

  edgeRange() : edges(parlay::make_slice<indexType*, indexType*>(nullptr, nullptr)) {}

  edgeRange(indexType* start, indexType* end, indexType id)
    : edges(parlay::make_slice<indexType*, indexType*>(start,end)), id_(id) {
    maxDeg = edges.size() - 1;
  }

  indexType operator [] (indexType j) const {
    if (j >= edges[0]) {
      std::cout << "ERROR: index exceeds degree while accessing neighbors" << std::endl;
      throw std::logic_error("ParlayANN invariant failed: utils/graph.h");
    } else return edges[j+1];
  }

  void append_neighbor(indexType nbh){
    if (edges[0] == maxDeg) {
      std::cout << "ERROR in append_neighbor: cannot exceed max degree "
                << maxDeg << std::endl;
      throw std::logic_error("ParlayANN invariant failed: utils/graph.h");
    } else {
      edges[edges[0]+1] = nbh;
      edges[0] += 1;
    }
  }

  template<typename rangeType>
  void update_neighbors(const rangeType& r){
    if (r.size() > maxDeg) {
      std::cout << "ERROR in update_neighbors: cannot exceed max degree "
                << maxDeg << std::endl;
      throw std::logic_error("ParlayANN invariant failed: utils/graph.h");
    }
    edges[0] = r.size();
    for (int i = 0; i < r.size(); i++) {
      edges[i+1] = r[i];
    }
  }

  template<typename rangeType>
  void append_neighbors(const rangeType& r){
    if (r.size() + edges[0] > maxDeg) {
      std::cout << "ERROR in append_neighbors for point " << id_
                << ": cannot exceed max degree " << maxDeg << std::endl;
      std::cout << edges[0] << std::endl;
      std::cout << r.size() << std::endl;
      throw std::logic_error("ParlayANN invariant failed: utils/graph.h");
    }
    for (int i = 0; i < r.size(); i++) {
      edges[edges[0] + i + 1] = r[i];
    }
    edges[0] += r.size();
  }

  void clear_neighbors(){
    edges[0] = 0;
  }

  void prefetch() const {
    int l = ((edges[0] + 1) * sizeof(indexType))/64;
    for (int i = 0; i < l; i++)
      __builtin_prefetch((char*) edges.begin() + i *  64);
  }

  template<typename F>
  void sort(F&& less){
    std::sort(edges.begin() + 1, edges.begin() + 1 + edges[0], less);}

  indexType* begin() const {return edges.begin() + 1;}

  indexType* end() const {return edges.begin() + 1 + edges[0];}

private:
  parlay::slice<indexType*, indexType*> edges;
  long maxDeg;
  indexType id_;
};

template<typename indexType_>
struct Graph{
  using indexType = indexType_;
  
  long max_degree() const {return maxDeg;}
  size_t size() const {return n;}

  Graph(){}

  void allocate_graph(long maxDeg, size_t n) {
    stride = static_cast<size_t>(maxDeg) + 1;
    if (maxDeg < 0 || n >= UINT32_MAX) throw std::invalid_argument("invalid graph capacity");
    const size_t cnt = checked_mul(n, checked_add(static_cast<size_t>(maxDeg), 1));
    const size_t num_bytes = checked_mul(cnt, sizeof(indexType));
    indexType* ptr = static_cast<indexType*>(allocate_aligned(num_bytes));
    parlay::parallel_for(0, cnt, [&] (long i) {ptr[i] = 0;});
    graph = std::shared_ptr<indexType[]>(ptr, std::free);
  }

  Graph(long maxDeg, size_t n) : maxDeg(maxDeg), n(n) {
    allocate_graph(maxDeg, n);
  }

  // Borrow degree+neighbors within fixed-stride caller-owned rows. The caller
  // keeps the allocation alive and exclusively borrowed until all jobs join.
  Graph(long maxDeg, size_t n, indexType* rows, size_t row_stride)
      : n(n), maxDeg(maxDeg), graph(rows, [](indexType*) {}), stride(row_stride) {
    if (!rows || maxDeg < 0 || row_stride < static_cast<size_t>(maxDeg) + 1)
      throw std::invalid_argument("invalid borrowed graph layout");
  }

  Graph(char* gFile) {
    static_assert(sizeof(indexType) == 4, "graph files use u32 IDs");
    std::ifstream reader(gFile, std::ios::binary | std::ios::ate);
    if (!reader) throw std::runtime_error("cannot open graph file");
    const auto length = reader.tellg();
    reader.seekg(0);
    n = read_u32(reader);
    maxDeg = read_u32(reader);
    if (!n || n >= UINT32_MAX || static_cast<size_t>(maxDeg) > n)
      throw std::runtime_error("invalid graph header");

    // Check degrees and total length before allocating the graph. A second
    // stream reads degrees while the first consumes edges: no N-sized offsets.
    size_t edge_count = 0;
    for (size_t i = 0; i < n; ++i) {
      const uint32_t degree = read_u32(reader);
      if (degree > maxDeg) throw std::runtime_error("graph degree exceeds capacity");
      edge_count = checked_add(edge_count, degree);
    }
    const size_t edge_offset = checked_add(8, checked_mul(n, 4));
    if (length < 0 || static_cast<uint64_t>(length) !=
        checked_add(edge_offset, checked_mul(edge_count, 4)))
      throw std::runtime_error("graph length mismatch");

    allocate_graph(maxDeg, n);
    std::ifstream degrees(gFile, std::ios::binary);
    degrees.seekg(8);
    for (size_t i = 0; i < n; ++i) {
      auto edges = (*this)[i];
      const uint32_t degree = read_u32(degrees);
      for (uint32_t j = 0; j < degree; ++j) {
        const uint32_t id = read_u32(reader);
        if (id >= n) throw std::runtime_error("graph ID out of range");
        edges.append_neighbor(id);
      }
    }
  }

  void save(char* oFile) {
    static_assert(sizeof(indexType) == 4, "graph files use u32 IDs");
    AtomicOutput output(oFile);
    output.word(static_cast<uint32_t>(n));
    output.word(static_cast<uint32_t>(maxDeg));
    for (size_t i = 0; i < n; ++i) output.word((*this)[i].size());
    for (size_t i = 0; i < n; ++i)
      for (auto id : (*this)[i]) output.word(id);
    output.publish();
  }

  edgeRange<indexType> operator [] (indexType i) const {
    if (i >= n) {
      std::cout << "ERROR: graph index out of range: " << i << std::endl;
      throw std::logic_error("ParlayANN invariant failed: utils/graph.h");
    }
    return edgeRange<indexType>(graph.get() + i * stride,
                                graph.get() + i * stride + maxDeg + 1,
                                i);
  }

  ~Graph(){}

private:
  size_t n;
  long maxDeg;
  std::shared_ptr<indexType[]> graph;
  size_t stride = 0;
};

} // end namespace
#endif // flexible
