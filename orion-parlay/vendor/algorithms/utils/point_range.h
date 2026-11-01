// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
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

#pragma once

#include <sys/mman.h>
#include <algorithm>
#include <climits>
#include <iostream>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"
#include "types.h"
#include "large_io.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace parlayANN {

template<class Point_>
struct PointRange{
  //using T = T_;
  using Point = Point_;
  using parameters = typename Point::parameters;
  using byte = uint8_t;

  long dimension() const {return params.dims;}
  //long aligned_dimension() const {return aligned_dims;}

  PointRange() : values(std::shared_ptr<byte[]>(nullptr, std::free)), n(0) {}

  template <typename PR>
  PointRange(const PR& pr, const parameters& p) : params(p)  {
    n = pr.size();
    int num_bytes = p.num_bytes();
    if (num_bytes <= 0) throw std::invalid_argument("invalid point size");
    aligned_bytes = (num_bytes <= 32) ? 32 : align_bytes(static_cast<size_t>(num_bytes), 64);
    size_t total_bytes = checked_mul(n, static_cast<size_t>(aligned_bytes));
    byte* ptr = static_cast<byte*>(allocate_aligned(total_bytes));
    values = std::shared_ptr<byte[]>(ptr, std::free);
    byte* vptr = values.get();
    parlay::parallel_for(0, n, [&] (long i) {
      Point::translate_point(vptr + i * aligned_bytes, pr[i], params);});
  }

  template <typename PR>
  PointRange (PR& pr) : PointRange(pr, Point::generate_parameters(pr)) { }

  template <typename PR>
  PointRange (PR& pr, int dims) : PointRange(pr, Point::generate_parameters(dims)) { }

  PointRange(char* filename) : values(std::shared_ptr<byte[]>(nullptr, std::free)){
      if(filename == NULL) {
        n = 0;
        return;
      }
      // Validate the entire file before reserving its resident allocation.
      // Point::parameters uses int byte counts, so reject overflowing dimensions.
      std::ifstream reader(filename, std::ios::binary);
      const auto shape = inspect_binary(filename, sizeof(typename Point::coordinate_type));
      if (shape.dimension > static_cast<size_t>(INT_MAX) / sizeof(typename Point::coordinate_type))
        throw std::overflow_error("point dimension exceeds parameter range");
      n = shape.count;
      params = parameters(shape.dimension);
      const size_t num_bytes = params.num_bytes();
      aligned_bytes = align_bytes(num_bytes, 64);
      const size_t total_bytes = checked_mul(n, static_cast<size_t>(aligned_bytes));
      values = std::shared_ptr<byte[]>(static_cast<byte*>(allocate_aligned(total_bytes)), std::free);
      reader.seekg(8);

      // Bound scratch by bytes, not a million vectors of arbitrary dimension.
      const size_t block_rows = std::max(size_t(1), size_t(8 * 1024 * 1024) / num_bytes);
      std::vector<byte> scratch(checked_mul(std::min(n, block_rows), num_bytes));
      for (size_t first = 0; first < n;) {
        const size_t last = first + std::min(block_rows, n - first);
        read_exact(reader, scratch.data(), checked_mul(last - first, num_bytes));
        parlay::parallel_for(first, last, [&](size_t i) {
          std::memcpy(values.get() + i * aligned_bytes,
                      scratch.data() + (i - first) * num_bytes, num_bytes);
        });
        first = last;
      }
  }

  size_t size() const { return n; }

  unsigned int get_dims() const { return params.dims; }
  
  Point operator [] (long i) const {
    if (i < 0 || static_cast<size_t>(i) >= n) {
      throw std::out_of_range("point index out of range");
    }
    return Point(values.get()+i*aligned_bytes, i, params);
  }

  byte* location(long i) const {
    return values.get() + i * aligned_bytes;
  }
  
  parameters params;

private:
  std::shared_ptr<byte[]> values;
  long aligned_bytes;
  size_t n;
};

} // end namespace
