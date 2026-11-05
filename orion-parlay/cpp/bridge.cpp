/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../vendor/algorithms/utils/euclidian_point.h"
#include "../vendor/algorithms/utils/mips_point.h"
#include "../vendor/algorithms/utils/staged_export.h"
#include "../vendor/algorithms/vamana/index.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace parlayANN {
std::atomic<size_t> g_total_i8_cmps{0}, g_total_i8_queries{0};
}

#ifdef ORION_REFERENCE_TESTS
#include "staged_reference.h"
#endif

// This ABI contains only C layouts. No C++ allocation/exception crosses it.
struct Options {
  size_t count, dimension, stride;
  uint32_t degree, build_l, max_extra, threads, local_pct, passes, metric;
  double alpha;
};
struct Failure {
  uint32_t code;
  char message[1024];
};
struct Partition {
  const uint32_t *local, *remote, *extra;
  size_t local_len, remote_len, extra_len;
};

template <class F> int boundary(Failure *error, F &&f) noexcept {
  *error = {};
  try {
    f();
    return 0;
  } catch (const std::invalid_argument &e) {
    error->code = 1;
    std::strncpy(error->message, e.what(), 1023);
  } catch (const std::bad_alloc &e) {
    error->code = 2;
    std::strncpy(error->message, e.what(), 1023);
  } catch (const std::overflow_error &e) {
    error->code = 1;
    std::strncpy(error->message, e.what(), 1023);
  } catch (const std::exception &e) {
    error->code = 4;
    std::strncpy(error->message, e.what(), 1023);
  } catch (...) {
    error->code = 4;
    std::strcpy(error->message, "unknown ParlayANN exception");
  }
  return -1;
}

// Existing point kernels accept mutable pointers, but construction only calls
// distance/prefetch on these views. They never normalize or retain Rust rows.
template <class P> struct BorrowedPoints {
  using Point = P;
  const uint8_t *bytes;
  size_t count, stride_bytes;
  typename P::parameters params;
  size_t size() const { return count; }
  long dimension() const { return params.dims; }
  unsigned get_dims() const { return params.dims; }
  P operator[](long i) const {
    if (i < 0 || static_cast<size_t>(i) >= count)
      throw std::out_of_range("point index");
    return P(const_cast<uint8_t *>(bytes + i * stride_bytes), i, params);
  }
};

// The owning baseline and borrowed-slab path run identical Vamana code.
template <class Points>
void construct(parlayANN::Graph<uint32_t> &graph, Points &points,
               parlayANN::ExtraStorage<uint32_t> &extras, const Options &o) {
  if (parlay::num_workers() != o.threads)
    throw std::logic_error("scheduler thread count mismatch");
  parlayANN::BuildParams bp(static_cast<long>(o.degree),
                            static_cast<long>(o.build_l), o.alpha,
                            static_cast<int>(o.passes));
  bp.max_extra = o.max_extra;
  parlayANN::knn_index<Points, Points, uint32_t> index(bp);
  parlayANN::stats<uint32_t> stats(o.count);
  index.extras = std::move(extras);
  index.build_index(graph, points, points, stats);
  extras = std::move(index.extras);
}

struct Handle {
  virtual ~Handle() = default;
  virtual bool next(Partition &) = 0;
  virtual void save(const char *, bool) = 0;
};

template <class P> struct Builder final : Handle {
  using PR = BorrowedPoints<P>;
  Options options;
  PR points;
  parlayANN::Graph<uint32_t> graph;
  parlayANN::ExtraStorage<uint32_t> extras;
  std::vector<parlayANN::NodeParts> batch;
  size_t emitted = 0, batch_first = 0;

  Builder(const void *data, const Options &o)
      : options(o), points{static_cast<const uint8_t *>(data), o.count,
                           o.stride * sizeof(typename P::coordinate_type),
                           typename P::parameters(o.dimension)},
        graph(o.degree, o.count) {
    construct(graph, points, extras, o);
  }

  bool next(Partition &out) override {
    if (emitted == options.count)
      return false;
    if (batch.empty() || emitted == batch_first + batch.size()) {
      batch_first = emitted;
      // Rust's mutable call ends all previous partition borrows. Release their
      // edge vectors before allocating the next batch, limiting transient
      // memory.
      batch.clear();
      parlay::execute_with_scheduler(options.threads, [&] {
        batch = parlayANN::partition_batch(
            graph, points, extras, options.local_pct, options.max_extra,
            emitted, std::min(options.count, emitted + size_t(65536)));
      });
    }
    const auto &p = batch[emitted++ - batch_first];
    out = {p.local.data(), p.remote.data(), p.extra.data(),
           p.local.size(), p.remote.size(), p.extra.size()};
    return true;
  }
  void save(const char *path, bool reference) override {
    parlay::execute_with_scheduler(options.threads, [&] {
#ifdef ORION_REFERENCE_TESTS
      if (reference) {
        parlayANN::save_staged_reference<P>(graph, points, extras,
                                            options.local_pct,
                                            options.max_extra, path, 0);
        return;
      }
#endif
      parlayANN::save_staged_format<P>(graph, points, extras, options.local_pct,
                                       options.max_extra, path, 0);
    });
  }
};

// Temporary degree lives at header[3], immediately before neighbors, so the
// original edgeRange implementation can address the Rust slab without copying.
// Extra candidates occupy fixed capacity R during building, then compact to the
// actual degree during finalization. No search can observe these temporary
// rows.
template <class P>
void build_slab(const void *data, const Options &o, uint32_t *slab,
                size_t stride) {
  BorrowedPoints<P> points{static_cast<const uint8_t *>(data), o.count,
                           o.stride * sizeof(typename P::coordinate_type),
                           typename P::parameters(o.dimension)};
  parlay::parallel_for(0, o.count, [&](size_t i) {
    std::fill_n(slab + i * stride, 4, uint32_t(0));
  });
  parlayANN::Graph<uint32_t> graph(o.degree, o.count, slab + 3, stride);
  parlayANN::ExtraStorage<uint32_t> extras;
  // Match the owning path when extras are disabled: no empty per-node views.
  if (o.max_extra != 0)
    extras = parlayANN::ExtraStorage<uint32_t>(
        o.count, o.max_extra, slab + 4 + o.degree, slab + 1, stride);
  construct(graph, points, extras, o);
  // Construction has joined. Each task reads all its source IDs into bounded
  // node scratch before overwriting that row; no other row's edges are read.
  parlay::parallel_for(0, o.count, [&](size_t i) {
    auto p = parlayANN::partition_node(graph, points, extras, o.local_pct,
                                       o.max_extra, i);
    auto row = slab + i * stride;
    auto end = std::copy(p.local.begin(), p.local.end(), row + 4);
    end = std::copy(p.remote.begin(), p.remote.end(), end);
    end = std::copy(p.extra.begin(), p.extra.end(), end);
    std::fill(end, row + stride, uint32_t(0));
    row[0] = p.local.size() + p.remote.size();
    row[1] = p.extra.size();
    row[2] = p.local.size();
    row[3] = 0;
  });
}

void validate(const Options &o, bool bytes) {
  if (!o.count || o.count > INT32_MAX || !o.dimension ||
      o.dimension > INT32_MAX / 4 || o.stride < o.dimension ||
      o.stride > SIZE_MAX / o.count / (bytes ? 1 : 4) || !o.degree ||
      o.degree > INT32_MAX || o.build_l < o.degree || o.build_l > INT32_MAX ||
      o.max_extra > INT32_MAX || !o.threads || o.threads > 4096 ||
      !o.local_pct || o.local_pct > 100 || !o.passes || o.passes > 16 ||
      !std::isfinite(o.alpha) || o.alpha < 1 || o.metric > 1 ||
      (bytes && o.metric != 0))
    throw std::invalid_argument(
        "invalid ParlayANN shape, metric or build options");
  // The original byte distance kernel accumulates into int32_t.
  if (bytes && o.dimension > size_t(INT32_MAX) / (255 * 255))
    throw std::invalid_argument(
        "u8 dimension exceeds int32 distance accumulator range");
}
extern "C" {
int orion_pa_build(const void *data, bool bytes, const Options *o,
                   Handle **result, Failure *error) noexcept {
  *result = nullptr;
  return boundary(error, [&] {
    validate(*o, bytes);
    if (!data)
      throw std::invalid_argument("null base pointer");
    std::unique_ptr<Handle> handle;
    parlay::execute_with_scheduler(o->threads, [&] {
      if (bytes)
        handle = std::make_unique<Builder<parlayANN::Euclidian_Point<uint8_t>>>(
            data, *o);
      else if (o->metric == 0)
        handle = std::make_unique<Builder<parlayANN::Euclidian_Point<float>>>(
            data, *o);
      else
        handle =
            std::make_unique<Builder<parlayANN::Mips_Point<float>>>(data, *o);
    });
    *result = handle.release();
  });
}
int orion_pa_build_into(const void *data, bool bytes, const Options *o,
                        uint32_t *slab, size_t words, size_t stride,
                        Failure *error) noexcept {
  return boundary(error, [&] {
    validate(*o, bytes);
    if (!data || !slab || stride < uint64_t(4) + o->degree + o->max_extra ||
        stride > words / o->count)
      throw std::invalid_argument("invalid slab length or stride");
    parlay::execute_with_scheduler(o->threads, [&] {
      if (bytes)
        build_slab<parlayANN::Euclidian_Point<uint8_t>>(data, *o, slab, stride);
      else if (o->metric == 0)
        build_slab<parlayANN::Euclidian_Point<float>>(data, *o, slab, stride);
      else
        build_slab<parlayANN::Mips_Point<float>>(data, *o, slab, stride);
    });
  });
}
int orion_pa_next(Handle *handle, Partition *part, bool *present,
                  Failure *error) noexcept {
  return boundary(error, [&] { *present = handle->next(*part); });
}
int orion_pa_save(Handle *handle, const char *path, bool reference,
                  Failure *error) noexcept {
  return boundary(error, [&] { handle->save(path, reference); });
}
void orion_pa_free(Handle *handle) noexcept { delete handle; }
// Exercised by Rust tests: failures on worker threads must reach the caller.
int orion_pa_test_exception(Failure *error) noexcept {
  return boundary(error, [&] {
    parlay::execute_with_scheduler(4, [] {
      parlay::parallel_for(
          0, 128,
          [](size_t i) {
            if (i == 65)
              throw std::runtime_error("worker failure probe");
          },
          1);
    });
  });
}
}
