#pragma once

#include "large_io.h"

namespace parlayANN {

struct BuildMemory {
  size_t base, graph, extras, statistics, permutations;
  size_t total() const {
    return checked_add(
        checked_add(base, graph),
        checked_add(extras, checked_add(statistics, permutations)));
  }
};

// Account only for known resident allocations. Batch search/pruning scratch,
// allocator overhead, optional quantized representations and page cache are
// extra.
inline BuildMemory build_memory(size_t n, size_t dimension,
                                size_t coordinate_bytes, size_t degree,
                                size_t extra_capacity) {
  const size_t base_stride =
      align_bytes(checked_mul(dimension, coordinate_bytes), 64);
  return {checked_mul(n, base_stride),
          checked_mul(n, checked_mul(checked_add(degree, 1), 4)),
          extra_capacity
              ? checked_mul(n, checked_add(checked_mul(extra_capacity, 4), 4))
              : 0,
          checked_mul(n, 8), checked_mul(n, 12)};
}

} // namespace parlayANN
