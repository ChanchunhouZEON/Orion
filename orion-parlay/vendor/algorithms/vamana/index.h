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

#include <math.h>

#include <algorithm>
#include <random>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../utils/point_range.h"
#include "../utils/extra_storage.h"
#include "../utils/graph.h"
#include "../utils/types.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/delayed.h"
#include "parlay/random.h"
#include "../utils/beamSearch.h"

namespace parlayANN {

template<typename PointRange, typename QPointRange, typename indexType>
struct knn_index {
  using Point = typename PointRange::Point;
  using QPoint = typename QPointRange::Point;
  using distanceType = typename Point::distanceType;
  using pid = std::pair<indexType, distanceType>;
  using PR = PointRange;
  using QPR = QPointRange;
  using GraphI = Graph<indexType>;

  BuildParams BP;
  std::set<indexType> delete_set;
  indexType start_point;

  /// StagedDiskANN-style per-node candidate residuals: candidates that were
  /// close enough to survive α-pruning but got cut because we hit R neighbors
  /// first. Populated only when BP.max_extra > 0 and only during the final
  /// α-pass of the forward insertion (beam-search visited → robustPrune site).
  ExtraStorage<indexType> extras;

  knn_index(BuildParams &BP) : BP(BP) {}

  indexType get_start() { return start_point; }

  //robustPrune routine as found in DiskANN paper, with the exception
  //that the new candidate set is added to the field new_nbhs instead
  //of directly replacing the out_nbh of p.
  //
  //If `extras_out` is non-null, after choosing the R neighbors it writes up
  //to `max_extra` remaining un-pruned candidates (first != -1, != p) sorted
  //by distance ascending. These are the StagedDiskANN `extra` zone.
  std::pair<parlay::sequence<indexType>, long>
  robustPrune(indexType p, parlay::sequence<pid>& cand,
              GraphI &G, PR &Points, double alpha, bool add = true,
              std::vector<indexType>* extras_out = nullptr,
              size_t max_extra = 0) {
    // add out neighbors of p to the candidate set.
    size_t out_size = G[p].size();
    std::vector<pid> candidates;
    long distance_comps = 0;
    for (auto x : cand) candidates.push_back(x);

    if(add){
      for (size_t i=0; i<out_size; i++) {
        distance_comps++;
        candidates.push_back(std::make_pair(G[p][i], Points[G[p][i]].distance(Points[p])));
      }
    }

    // Sort the candidate set according to distance from p
    auto less = [&](std::pair<indexType, distanceType> a, std::pair<indexType, distanceType> b) {
      return a.second < b.second || (a.second == b.second && a.first < b.first);
    };
    std::sort(candidates.begin(), candidates.end(), less);

    // remove any duplicates
    auto new_end =std::unique(candidates.begin(), candidates.end(),
			      [&] (auto x, auto y) {return x.first == y.first;});
    candidates = std::vector(candidates.begin(), new_end);

    std::vector<indexType> new_nbhs;
    new_nbhs.reserve(BP.R);

    // Bounded **sorted-by-dist-asc** scratch for extras (when capture
    // is enabled). Holds at most `max_extra` `(dist_to_p, id)` pairs;
    // any j-loop occluded candidate closer than the current worst
    // entry kicks out the worst. This keeps extras = "the K closest
    // candidates that occlusion threw away", which can compete with
    // remote edges in the export-time dist-merge (unlike leftover
    // [R..end] candidates which are by construction farther than
    // every remote edge).
    std::vector<std::pair<distanceType, indexType>> extras_sorted;
    if (extras_out && max_extra > 0) {
      extras_sorted.reserve(max_extra + 1);
    }
    auto try_insert_extra = [&](distanceType d, indexType id) {
      if (extras_sorted.size() < max_extra) {
        auto pos = std::upper_bound(
            extras_sorted.begin(), extras_sorted.end(),
            std::make_pair(d, id),
            [](const auto& a, const auto& b){ return a.first < b.first; });
        extras_sorted.insert(pos, {d, id});
      } else if (d < extras_sorted.back().first) {
        extras_sorted.pop_back();
        auto pos = std::upper_bound(
            extras_sorted.begin(), extras_sorted.end(),
            std::make_pair(d, id),
            [](const auto& a, const auto& b){ return a.first < b.first; });
        extras_sorted.insert(pos, {d, id});
      }
    };

    size_t candidate_idx = 0;

    while (new_nbhs.size() < BP.R && candidate_idx < candidates.size()) {
      // Don't need to do modifications.
      int p_star = candidates[candidate_idx].first;
      candidate_idx++;
      if (p_star == p || p_star == -1) {
        continue;
      }

      new_nbhs.push_back(p_star);

      for (size_t i = candidate_idx; i < candidates.size(); i++) {
        int p_prime = candidates[i].first;
        if (p_prime != -1) {
          distance_comps++;
          distanceType dist_starprime = Points[p_star].distance(Points[p_prime]);
          distanceType dist_pprime = candidates[i].second;
          if (alpha * dist_starprime <= dist_pprime) {
            // **Capture** the occluded candidate before marking -1.
            // Its dist_pprime (= dist to p) is in the "could-have-been-
            // top-R" range, so it has a real chance of beating a
            // remote edge in the export-time merge.
            if (extras_out && max_extra > 0
                && p_prime != static_cast<int>(p)) {
              try_insert_extra(dist_pprime, static_cast<indexType>(p_prime));
            }
            candidates[i].first = -1;
          }
        }
      }
    }

    // Strip distances; export-time partition recomputes them from
    // Points. extras_out receives ID-only, distance-asc order.
    if (extras_out) {
      extras_out->clear();
      extras_out->reserve(extras_sorted.size());
      for (auto& [d, id] : extras_sorted) {
        (void)d;
        extras_out->push_back(id);
      }
    }

    auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
    return std::pair(new_neighbors_seq, distance_comps);
  }

  //wrapper to allow calling robustPrune on a sequence of candidates
  //that do not come with precomputed distances
  std::pair<parlay::sequence<indexType>, long>
  robustPrune(indexType p, parlay::sequence<indexType> candidates,
              GraphI &G, PR &Points, double alpha, bool add = true){

    parlay::sequence<pid> cc;
    long distance_comps = 0;
    cc.reserve(candidates.size()); // + size_of(p->out_nbh));
    for (size_t i=0; i<candidates.size(); ++i) {
      distance_comps++;
      cc.push_back(std::make_pair(candidates[i], Points[candidates[i]].distance(Points[p])));
    }
    auto [ngh_seq, dc] = robustPrune(p, cc, G, Points, alpha, add);
    return std::pair(ngh_seq, dc + distance_comps);
  }

  // add ngh to candidates without adding any repeats
  template<typename rangeType1, typename rangeType2>
  void add_neighbors_without_repeats(const rangeType1 &ngh, rangeType2& candidates) {
    std::unordered_set<indexType> a;
    for (auto c : candidates) a.insert(c);
    for (int i=0; i < ngh.size(); i++)
      if (a.count(ngh[i]) == 0) candidates.push_back(ngh[i]);
  }

  void set_start(){start_point = 0;}

  void build_index(GraphI &G, PR &Points, QPR &QPoints,
                   stats<indexType> &BuildStats, bool sort_neighbors = true){
    if (BP.verbose) std::cout << "Building graph..." << std::endl;
    set_start();
    parlay::sequence<indexType> inserts = parlay::tabulate(Points.size(), [&] (size_t i){
      return static_cast<indexType>(i);});
    if (BP.single_batch != 0) {
      int degree = BP.single_batch;
      std::cout << "Using single batch per round with " << degree << " random start edges" << std::endl;
      parlay::random_generator gen;
      std::uniform_int_distribution<long> dis(0, G.size());
      parlay::parallel_for(0, G.size(), [&] (long i) {
        std::vector<indexType> outEdges(degree);
        for (int j = 0; j < degree; j++) {
          auto r = gen[i*degree + j];
          outEdges[j] = dis(r);
        }
        G[i].update_neighbors(outEdges);
      });
    }

    // Allocate per-node extras buffer only if the caller requested them.
    // BP.max_extra is the configured hard cap; pruning preserves distance order.
    bool capture_extras = (BP.max_extra > 0);
    if (capture_extras) {
      if (extras.size() == 0)
        extras = ExtraStorage<indexType>(G.size(), static_cast<size_t>(BP.max_extra));
      else if (extras.size() != G.size() || extras.capacity() != BP.max_extra)
        throw std::invalid_argument("provided extras storage has wrong shape");
    }

    if (BP.verbose) std::cout << "number of passes = " << BP.num_passes << std::endl;
    for (int i=0; i < BP.num_passes; i++) {
      bool is_last_pass = (i == BP.num_passes - 1);
      double pass_alpha = is_last_pass ? BP.alpha : 1.0;
      // Capture extras only on the final α-pass — that's when the
      // graph state most resembles what the search will see, so
      // pruned-candidate residuals are most representative. Earlier
      // passes' candidates are noisy (graph still saturating).
      bool capture_this_pass = capture_extras && is_last_pass;
      batch_insert(inserts, G, Points, QPoints, BuildStats,
                   pass_alpha, true, 2, .02, BP.verbose, capture_this_pass);
    }

    if (sort_neighbors) {
      parlay::parallel_for (0, G.size(), [&] (long i) {
        auto less = [&] (indexType j, indexType k) {
          return Points[i].distance(Points[j]) < Points[i].distance(Points[k]);};
        G[i].sort(less);});
    }
    // Note: extras finalisation (dedup against G[i] + merge with
    // remote by distance + split into local/extra) is handled by
    // `save_staged_format` (v3) at export time, not here. This way
    // the same `extras[i]` data can drive the partition logic and
    // any other downstream consumers without re-doing work.
  }

  void batch_insert(parlay::sequence<indexType> &inserts,
                    GraphI &G, PR &Points, QPR &QPoints,
                    stats<indexType> &BuildStats, double alpha,
                    bool random_order = false, double base = 2,
                    double max_fraction = .02, bool print=true,
                    bool capture_extras = false) {
    for(int p : inserts){
      if(p < 0 || p > (int) G.size()){
        std::cout << "ERROR: invalid point "
                  << p << " given to batch_insert" << std::endl;
        throw std::logic_error("ParlayANN invariant failed: vamana/index.h");
      }
    }
    size_t n = G.size();
    size_t m = inserts.size();
    size_t inc = 0;
    size_t count = 0;
    float frac = 0.0;
    float progress_inc = .1;
    size_t max_batch_size = std::min(static_cast<size_t>(max_fraction * static_cast<float>(n)),
                                     BP.build_batch_size);
    //fix bug where max batch size could be set to zero
    if(max_batch_size == 0) max_batch_size = std::min(n, BP.build_batch_size);
    parlay::sequence<int> rperm;
    if (random_order) 
      rperm = parlay::random_permutation<int>(static_cast<int>(m));
    else
      rperm = parlay::tabulate(m, [&](int i) { return i; });
    auto shuffled_inserts =
      parlay::tabulate(m, [&](size_t i) { return inserts[rperm[i]]; });
    parlay::internal::timer t_beam("beam search time");
    parlay::internal::timer t_bidirect("bidirect time");
    parlay::internal::timer t_prune("prune time");
    t_beam.stop();
    t_bidirect.stop();
    t_prune.stop();
    while (count < m) {
      size_t floor;
      size_t ceiling;
      if (pow(base, inc) <= max_batch_size) {
        floor = static_cast<size_t>(pow(base, inc)) - 1;
        ceiling = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
        count = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
      } else {
        floor = count;
        ceiling = std::min(count + static_cast<size_t>(max_batch_size), m);
        count += static_cast<size_t>(max_batch_size);
      }

      if (BP.single_batch != 0) {
        floor = 0;
        ceiling = m;
        count = m;
      }

      parlay::sequence<parlay::sequence<indexType>> new_out_(ceiling-floor);
      // search for each node starting from the start_point, then call
      // robustPrune with the visited list as its candidate set
      t_beam.start();

      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        size_t index = shuffled_inserts[i];
        indexType sp = BP.single_batch ? i : start_point;
        parlay::sequence<indexType> starting_points = {sp};
        QueryParams QP(0, BP.L, (long) Points.size(), (long) G.max_degree(),
                       BP.rerank_factor, BP.batch_factor);
        bool use_filtering = (Points.params.num_bytes() != QPoints.params.num_bytes());
        auto r = filtered_beam_search(G,
                                      Points[index],
                                      Points,
                                      QPoints[index],
                                      QPoints,
                                      starting_points,
                                      QP,
                                      use_filtering);
        auto visited = r.first.second;
        BuildStats.increment_dist(index, r.second);
        BuildStats.increment_visited(index, visited.size());

        long rp_distance_comps;
        if (capture_extras) {
          // visited is already a parlay::sequence<pid> (id, dist) from
          // filtered_beam_search; thread it straight into the pid-variant of
          // robustPrune with the extras_out pointer.
          std::vector<indexType> local_extras;
          std::tie(new_out_[i-floor], rp_distance_comps) =
              robustPrune(index, visited, G, Points, alpha, true,
                          &local_extras, static_cast<size_t>(BP.max_extra));
          // Last-pass-only capture (build_index gates this); a
          // straight assignment is fine — there's nothing earlier
          // to merge with.
          extras.assign(index, local_extras);
        } else {
          std::tie(new_out_[i-floor], rp_distance_comps) = robustPrune(index, visited, G, Points, alpha);
        }
        BuildStats.increment_dist(index, rp_distance_comps);
      });

      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        G[shuffled_inserts[i]].update_neighbors(new_out_[i-floor]);
      });

      t_beam.stop();

      // make each edge bidirectional by first adding each new edge
      //(i,j) to a sequence, then semisorting the sequence by key values
      t_bidirect.start();

      auto flattened = parlay::delayed::flatten(parlay::tabulate(ceiling - floor, [&](size_t i) {
        indexType index = shuffled_inserts[i + floor];
        return parlay::delayed::map(new_out_[i], [=] (indexType ngh) {
          return std::pair(ngh, index);});}));
      auto grouped_by = parlay::group_by_key(parlay::delayed::to_sequence(flattened));

      t_bidirect.stop();
      t_prune.start();
      // finally, add the bidirectional edges; if they do not make
      // the vertex exceed the degree bound, just add them to out_nbhs;
      // otherwise, use robustPrune on the vertex with user-specified alpha
      parlay::parallel_for(0, grouped_by.size(), [&](size_t j) {
        auto &[index, candidates] = grouped_by[j];
	size_t newsize = candidates.size() + G[index].size();
        if (newsize <= BP.R) {
	  add_neighbors_without_repeats(G[index], candidates);
	  G[index].update_neighbors(candidates);
        } else {
          auto [new_out_2_, distance_comps] = robustPrune(index, std::move(candidates), G, Points, alpha);
	  G[index].update_neighbors(new_out_2_);
          BuildStats.increment_dist(index, distance_comps);
        }
      });
      t_prune.stop();

      if (print && BP.single_batch == 0) {
        auto ind = frac * n;
        if (floor <= ind && ceiling > ind) {
          frac += progress_inc;
          std::cout << "Pass " << 100 * frac << "% complete"
                    << std::endl;
        }
      }
      inc += 1;
    }
    if (BP.verbose) { t_beam.total(); t_bidirect.total(); t_prune.total(); }
  }

};

} // end namespace
