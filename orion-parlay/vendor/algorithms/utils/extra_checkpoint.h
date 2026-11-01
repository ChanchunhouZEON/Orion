#pragma once

#include "extra_storage.h"
#include "graph.h"

namespace parlayANN {

// This is a graph binding / corruption checksum, not a cryptographic identity.
inline uint64_t checksum_word(uint64_t hash, uint32_t word) {
  for (int shift = 0; shift < 32; shift += 8) {
    hash ^= static_cast<unsigned char>(word >> shift);
    hash *= 1099511628211ULL;
  }
  return hash;
}

template <class Id> uint64_t graph_checksum(Graph<Id> &graph) {
  uint64_t hash = checksum_word(14695981039346656037ULL, graph.size());
  hash = checksum_word(hash, graph.max_degree());
  for (size_t i = 0; i < graph.size(); ++i) {
    hash = checksum_word(hash, graph[i].size());
    for (auto id : graph[i])
      hash = checksum_word(hash, id);
  }
  return hash;
}

template <class Id>
void save_extra_checkpoint(const char *path, Graph<Id> &graph,
                           const ExtraStorage<Id> &extras) {
  if (extras.size() != graph.size())
    throw std::runtime_error("missing captured extras");
  AtomicOutput output(path);
  const uint64_t binding = graph_checksum(graph);
  for (uint32_t word :
       {0x52545845u, 1u, static_cast<uint32_t>(graph.size()),
        static_cast<uint32_t>(extras.capacity()),
        static_cast<uint32_t>(binding), static_cast<uint32_t>(binding >> 32)})
    output.word(word);

  uint64_t checksum = 14695981039346656037ULL;
  for (size_t i = 0; i < extras.size(); ++i) {
    output.word(extras[i].size());
    checksum = checksum_word(checksum, extras[i].size());
    for (auto id : extras[i]) {
      output.word(id);
      checksum = checksum_word(checksum, id);
    }
  }
  output.word(static_cast<uint32_t>(checksum));
  output.word(static_cast<uint32_t>(checksum >> 32));
  output.publish();
}

template <class Id>
ExtraStorage<Id> load_extra_checkpoint(const char *path, Graph<Id> &graph,
                                       size_t capacity) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open extras checkpoint");
  const auto length = input.tellg();
  const size_t minimum = checked_add(32, checked_mul(graph.size(), 4));
  const size_t maximum =
      checked_add(minimum, checked_mul(checked_mul(graph.size(), capacity), 4));
  if (length < 0 || static_cast<uint64_t>(length) < minimum ||
      static_cast<uint64_t>(length) > maximum)
    throw std::runtime_error("extras checkpoint length mismatch");
  input.seekg(0);
  if (read_u32(input) != 0x52545845u || read_u32(input) != 1 ||
      read_u32(input) != graph.size() || read_u32(input) != capacity)
    throw std::runtime_error("extras checkpoint header mismatch");
  uint64_t binding = read_u32(input);
  binding |= uint64_t(read_u32(input)) << 32;
  if (binding != graph_checksum(graph))
    throw std::runtime_error("extras belong to another graph");

  ExtraStorage<Id> extras(graph.size(), capacity);
  std::vector<Id> row;
  row.reserve(capacity);
  uint64_t checksum = 14695981039346656037ULL;
  for (size_t i = 0; i < graph.size(); ++i) {
    const uint32_t count = read_u32(input);
    if (count > capacity)
      throw std::runtime_error("extras count exceeds capacity");
    checksum = checksum_word(checksum, count);
    row.clear();
    for (uint32_t j = 0; j < count; ++j) {
      const uint32_t id = read_u32(input);
      if (id >= graph.size())
        throw std::runtime_error("extras ID out of range");
      checksum = checksum_word(checksum, id);
      row.push_back(id);
    }
    extras.assign(i, row);
  }
  uint64_t saved = read_u32(input);
  saved |= uint64_t(read_u32(input)) << 32;
  if (saved != checksum || input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("extras checksum/trailing bytes mismatch");
  return extras;
}

} // namespace parlayANN
