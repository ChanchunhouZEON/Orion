#pragma once

#include "large_io.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"

namespace parlayANN {

// Each node owns a disjoint fixed-capacity slot. Parallel forward insertions
// may write different nodes without allocating one heap object per node.
template <class Id> class ExtraStorage {
public:
  ExtraStorage() = default;
  ExtraStorage(size_t nodes, size_t capacity)
      : nodes_(nodes), capacity_(capacity),
        ids_(parlay::sequence<Id>::uninitialized(slot_count(nodes, capacity))),
        counts_(capacity ? nodes : 0, 0) {}

  // Counts and IDs may occupy different fields in the same caller-owned slab.
  ExtraStorage(size_t nodes, size_t capacity, Id *ids, uint32_t *counts,
               size_t stride)
      : nodes_(nodes), capacity_(capacity), borrowed_ids_(ids),
        borrowed_counts_(counts), borrowed_stride_(stride) {
    if (!ids || !counts || stride < capacity)
      throw std::invalid_argument("invalid borrowed extras layout");
  }

  size_t size() const { return nodes_; }
  size_t capacity() const { return capacity_; }

  auto operator[](size_t node) const {
    if (node >= nodes_)
      throw std::out_of_range("extras node");
    // Avoid pointer arithmetic on an empty allocation when capacity is zero.
    const Id *begin = capacity_ ? row(node) : nullptr;
    const size_t count = capacity_ ? node_count(node) : 0;
    return parlay::make_slice(begin, count ? begin + count : begin);
  }

  template <class Range> void assign(size_t node, const Range &values) {
    if (node >= nodes_ || values.size() > capacity_)
      throw std::out_of_range("extras slot capacity or node");
    if (values.size())
      std::copy(values.begin(), values.end(), row(node));
    if (capacity_)
      node_count(node) = static_cast<uint32_t>(values.size());
  }

private:
  Id *row(size_t node) const {
    return borrowed_ids_ ? borrowed_ids_ + node * borrowed_stride_
                         : const_cast<Id *>(ids_.data()) + node * capacity_;
  }
  uint32_t &node_count(size_t node) const {
    return borrowed_counts_ ? borrowed_counts_[node * borrowed_stride_]
                            : const_cast<uint32_t &>(counts_[node]);
  }
  Id *borrowed_ids_ = nullptr;
  uint32_t *borrowed_counts_ = nullptr;
  size_t borrowed_stride_ = 0;
  static size_t slot_count(size_t nodes, size_t capacity) {
    if (nodes >= UINT32_MAX || capacity > UINT32_MAX)
      throw std::invalid_argument("extras dimensions exceed u32 IDs");
    return checked_mul(nodes, capacity);
  }

  size_t nodes_ = 0;
  size_t capacity_ = 0;
  parlay::sequence<Id> ids_;
  parlay::sequence<uint32_t> counts_;
};

} // namespace parlayANN
