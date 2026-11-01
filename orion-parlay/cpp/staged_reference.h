/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "../vendor/algorithms/utils/graph.h"
#include "../vendor/algorithms/utils/large_io.h"
#include "../vendor/algorithms/utils/point_range.h"
#include <unordered_set>
#include <vector>

namespace parlayANN {

// STAG v3: six little-endian u32 header words (magic, version, N, R,
// extra capacity, entry), then per node three counts and local/remote/extra
// IDs. Partition the distance-sorted edge/candidate union at floor(local_pct *
// GRAPH DEGREE / 100). Top graph edges are local, top candidates are extras;
// remaining graph edges are remote and remaining candidates are discarded.
// Keep the original distance-only comparator (including tie behavior).
template <typename Point, typename PointRange, typename indexType,
          typename Extras>
void save_staged_reference(Graph<indexType> &G, PointRange &Points,
                        const Extras &extras, int local_pct,
                        uint32_t max_extra_cap, const char *outfile,
                        uint32_t entry_point, size_t batch_nodes = 65536,
                        OutputHooks hooks = {}) {
  const size_t n = G.size();
  if (!n || n >= UINT32_MAX || Points.size() != n || entry_point >= n ||
      G.max_degree() < 0 || static_cast<size_t>(G.max_degree()) > UINT32_MAX ||
      local_pct < 1 || local_pct > 100 || !batch_nodes)
    throw std::invalid_argument("invalid STAG export dimensions/options");
  if (max_extra_cap && extras.size() != n)
    throw std::invalid_argument(
        "STAG export requires captured extras for every node");

  const uint32_t max_deg = static_cast<uint32_t>(G.max_degree());
  const bool have_extras_global = (extras.size() == n);
  const double X = static_cast<double>(local_pct) / 100.0;
  struct NodeParts {
    std::vector<uint32_t> local, remote, extra;
  };

  AtomicOutput output(outfile, hooks);
  for (uint32_t word : {0x53544147u, 3u, static_cast<uint32_t>(n), max_deg,
                        max_extra_cap, entry_point})
    output.word(word);

  uint64_t sum_local = 0, sum_remote = 0, sum_extra = 0;
  for (size_t first = 0; first < n;) {
    const size_t last = first + std::min(batch_nodes, n - first);
    std::vector<NodeParts> parts(last - first);

    // Validate before entering parallel work: exceptions must propagate on
    // the calling thread, and no invalid ID may reach Points[id].
    for (size_t i = first; i < last; ++i) {
      if (G[i].size() > max_deg)
        throw std::runtime_error("graph degree exceeds capacity");
      for (auto id : G[i])
        if (id >= n)
          throw std::runtime_error("graph ID out of range");
      if (have_extras_global) {
        if (extras[i].size() > max_extra_cap)
          throw std::runtime_error("extras exceed header capacity");
        for (auto id : extras[i])
          if (id >= n)
            throw std::runtime_error("extra ID out of range");
      }
    }
    parlay::parallel_for(first, last, [&](size_t i) {
      auto edges = G[i];
      uint32_t deg = edges.size();
      const uint32_t i_u32 = static_cast<uint32_t>(i);

      // Build a set of G[i] for fast lookup later.
      std::unordered_set<uint32_t> nbr_set;
      nbr_set.reserve(deg * 2);
      for (uint32_t k = 0; k < deg; ++k)
        nbr_set.insert(edges[k]);

      // Combined list: (dist, id, is_in_G).
      struct Entry {
        float d;
        uint32_t id;
        bool in_g;
      };
      std::vector<Entry> combined;
      combined.reserve(deg + (have_extras_global ? extras[i].size() : 0));

      // Add G[i] neighbours.
      for (uint32_t k = 0; k < deg; ++k) {
        uint32_t j = edges[k];
        combined.push_back({Points[i].distance(Points[j]), j, true});
      }
      // Add extras pool (filter dup against G[i] and self; dedup IDs).
      if (have_extras_global && extras[i].size() > 0) {
        std::unordered_set<uint32_t> seen;
        seen.reserve(extras[i].size() * 2);
        for (auto e : extras[i]) {
          uint32_t e32 = static_cast<uint32_t>(e);
          if (e32 == i_u32)
            continue;
          if (nbr_set.find(e32) != nbr_set.end())
            continue;
          if (!seen.insert(e32).second)
            continue;
          combined.push_back({Points[i].distance(Points[e32]), e32, false});
        }
      }

      std::sort(combined.begin(), combined.end(),
                [](const Entry &a, const Entry &b) { return a.d < b.d; });

      // Threshold: top X% of **the per-node graph degree** (not of
      // combined.size()). Anchoring to deg keeps |local + extra|
      // proportional to the graph density, independent of how
      // big the captured-extras pool happens to be.
      const size_t cutoff = static_cast<size_t>(X * static_cast<double>(deg));

      std::vector<uint32_t> local;
      std::vector<uint32_t> remote;
      std::vector<uint32_t> extra;
      local.reserve(cutoff);
      extra.reserve(cutoff);
      remote.reserve(combined.size() - cutoff);

      for (size_t k = 0; k < combined.size(); ++k) {
        const Entry &e = combined[k];
        if (k < cutoff) {
          if (e.in_g)
            local.push_back(e.id);
          else
            extra.push_back(e.id);
        } else {
          if (e.in_g)
            remote.push_back(e.id);
          // out-of-threshold extras (= pruned candidates that
          // didn't make it into top-X%): discard.
        }
      }

      // No truncation needed: capture buffer (robustPrune) was
      // already capped at max_extra, so |extras_pool| ≤ max_extra.
      // After partition the extra zone is at most that many items.
      parts[i - first].local = std::move(local);
      parts[i - first].remote = std::move(remote);
      parts[i - first].extra = std::move(extra);
    });

    // One bounded byte buffer per block avoids tiny fwrite calls per edge.
    // IDs are encoded explicitly, so the format is little-endian on all hosts.
    std::vector<unsigned char> encoded;
    auto append = [&](uint32_t value) {
      for (int shift = 0; shift < 32; shift += 8)
        encoded.push_back(static_cast<unsigned char>(value >> shift));
    };
    for (const auto &node : parts) {
      append(static_cast<uint32_t>(node.local.size()));
      append(static_cast<uint32_t>(node.remote.size()));
      append(static_cast<uint32_t>(node.extra.size()));
      for (auto id : node.local)
        append(id);
      for (auto id : node.remote)
        append(id);
      for (auto id : node.extra)
        append(id);
      sum_local += node.local.size();
      sum_remote += node.remote.size();
      sum_extra += node.extra.size();
    }
    output.write(encoded.data(), encoded.size());
    first = last;
  }
  output.publish();
  std::cout << "Staged export complete (v3). mean local="
            << double(sum_local) / n << " remote=" << double(sum_remote) / n
            << " extra=" << double(sum_extra) / n << std::endl;
}

} // namespace parlayANN
