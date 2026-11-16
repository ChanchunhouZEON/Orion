/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

// Private single-request adapter over the existing ParlayANN search templates.
// No copied search kernel. The two quantization levels mirror neighborsTime.C
// and vamana/neighbors.h; query conversion is inside each request, not preloaded.
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "algorithms/utils/point_range.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/jl_point.h"
#include "algorithms/utils/beamSearch.h"
namespace parlayANN {
std::atomic<size_t> g_total_i8_cmps{0}, g_total_i8_queries{0};
}
using namespace parlayANN;
static thread_local std::string error_message;

struct Index {
  virtual ~Index() = default;
  virtual void search(const float*, size_t, size_t, size_t, uint32_t*) const = 0;
};
template<class Point> struct Row {
  using P = Point;
  Point point;
  typename Point::parameters params;
  size_t size() const { return 1; }
  size_t dimension() const { return params.dims; }
  Point operator[](size_t) const { return point; }
};
template<class To, class From> To convert(From& source) {
  if constexpr (std::is_same_v<To, From>) return static_cast<const To&>(source);
  else return To(source);
}
template<class To, class From, class Parameters>
To convert_query(From& source, const Parameters& params) {
  if constexpr (std::is_same_v<To, From>) return static_cast<const To&>(source);
  else return To(source, params);
}
template<class To, class Base, class Admission>
To filter_base(Base& base, Admission& admission) {
  if constexpr (std::is_same_v<To, Admission>) return static_cast<const To&>(admission);
  else return convert<To>(base);
}
template<class To, class Base, class Admission, class Parameters>
To filter_query(Base& base, Admission& admission, const Parameters& params) {
  if constexpr (std::is_same_v<To, Admission>) return static_cast<const To&>(admission);
  else return convert_query<To>(base, params);
}
template<class Raw, class Exact> PointRange<Exact> load_base(char* path, bool normalize) {
  PointRange<Raw> raw(path);
  if (normalize) for (size_t i = 0; i < raw.size(); ++i) raw[i].normalize();
  return convert<PointRange<Exact>>(raw);
}

template<class Raw, class Exact, class Admission, class Filter>
struct Model final : Index {
  Graph<uint32_t> graph;
  PointRange<Exact> base;
  PointRange<Admission> admission;
  PointRange<Filter> filter;
  bool normalize;
  double rerank;
  Model(char* data, char* graph_path, bool norm, double factor)
      : graph(graph_path), base(load_base<Raw, Exact>(data, norm)),
        admission(convert<PointRange<Admission>>(base)),
        filter(filter_base<PointRange<Filter>>(base, admission)), normalize(norm), rerank(factor) {
    if (base.size() != graph.size()) throw std::invalid_argument("graph/base count mismatch");
  }
  void search(const float* values, size_t dimension, size_t k, size_t beam, uint32_t* output) const override {
    if (dimension != size_t(base.dimension()) || !k || k > graph.size() || beam < k)
      throw std::invalid_argument("invalid query shape/k/beam");
    // Conversion, scratch and result allocation belong to request latency.
    std::vector<float> copy(values, values + dimension);
    typename Raw::parameters params(dimension);
    Raw point(reinterpret_cast<uint8_t*>(copy.data()), 0, params);
    if (normalize) point.normalize();
    Row<Raw> row{point, params};
    PointRange<Exact> q(row, base.params);
    auto aq = convert_query<PointRange<Admission>>(q, admission.params);
    auto fq = filter_query<PointRange<Filter>>(q, aq, filter.params);
    stats<uint32_t> counters(0); // no shared per-query counter writes
    QueryParams options(k, beam, graph.size(), graph.max_degree(), rerank);
    auto results = beam_search_rerank(q[0], aq[0], fq[0], graph, base, admission, filter,
                                     counters, parlay::sequence<uint32_t>{0}, options, false);
    for (size_t i = 0; i < k; ++i) output[i] = results[i].first;
  }
};

template<class Raw, class Exact>
Index* make(char* base, char* graph, bool norm, int mode, double rerank) {
  if (mode == 0) return new Model<Raw, Exact, Exact, Exact>(base, graph, norm, rerank);
  if constexpr (Exact::is_metric) {
    using A = Euclidian_Point<uint8_t>;
    if (mode == 1) return new Model<Raw, Exact, A, A>(base, graph, norm, rerank);
    if (mode == 3) return new Model<Raw, Exact, A, Euclidean_JL_Sparse_Point<1024>>(base, graph, norm, rerank);
  } else {
    using A = Quantized_Mips_Point<8, true, 255>;
    if (mode == 1) return new Model<Raw, Exact, A, A>(base, graph, norm, rerank);
    if (mode == 5) return new Model<Raw, Exact, A, Mips_JL_Sparse_Point<1024>>(base, graph, norm, rerank);
  }
  throw std::invalid_argument("unsupported metric/quantize_mode combination");
}
// Foreign caller threads must not each create an eight-worker scheduler. A
// single ANN request is sequential; the Python harness provides concurrency.
extern "C" void latency_thread_init() {
  static thread_local parlay::internal::scheduler_type caller_scheduler(1);
}
extern "C" const char* latency_error() { return error_message.c_str(); }
// Pointer lifetimes are private to the Python adapter: close only after clients join.
extern "C" Index* latency_open(char* base, char* graph, int metric, int bits, int mode, double rerank) {
  try {
    if (!base || !graph || rerank < 1 || metric < 0 || metric > 2)
      throw std::invalid_argument("invalid ParlayANN configuration");
    if (metric == 0) {
      using R = Euclidian_Point<float>;
      if (bits == 0) return make<R, R>(base, graph, false, mode, rerank);
      if (bits == 8) return make<R, Euclidian_Point<uint8_t>>(base, graph, false, mode, rerank);
      if (bits == 16) return make<R, Euclidian_Point<uint16_t>>(base, graph, false, mode, rerank);
    } else {
      using R = Mips_Point<float>;
      if (bits == 0) return make<R, R>(base, graph, metric == 2, mode, rerank);
      if (bits == 8) return make<R, Quantized_Mips_Point<8>>(base, graph, metric == 2, mode, rerank);
      if (bits == 16) return make<R, Quantized_Mips_Point<16>>(base, graph, metric == 2, mode, rerank);
    }
    throw std::invalid_argument("quantize_bits must be 0, 8 or 16");
  } catch (const std::exception& e) { error_message = e.what(); return nullptr; }
}
extern "C" int latency_search(const Index* index, const float* query, size_t dimension,
                               size_t k, uint32_t beam, uint32_t* output) {
  try {
    if (!index || !query || !output) throw std::invalid_argument("null search buffer");
    index->search(query, dimension, k, beam, output);
    return 0;
  } catch (const std::exception& e) { error_message = e.what(); return -1; }
}
extern "C" void latency_close(Index* index) { delete index; }
