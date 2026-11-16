/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

// Private read-only control: the upstream search kernel is unchanged. Reserve
// before publishing the handle; each request leases an exclusive scratch slot.
#include <usearch/index_dense.hpp>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

using namespace unum::usearch;
static thread_local std::string last_error;

struct Control {
    index_dense_t index;
    std::shared_mutex configuration;
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<std::size_t> free_slots;

    Control(index_dense_t&& loaded, std::size_t slots) : index(std::move(loaded)) {
        // Reserve all contexts once, including query casts. No reserve, mutation
        // or automatic context resizing is allowed during search.
        index.reserve(index_limits_t(index.capacity(), slots));
        free_slots.reserve(slots);
        for (std::size_t i = 0; i < slots; ++i) free_slots.push_back(i);
    }

    struct Lease {
        Control& owner;
        std::size_t slot;
        explicit Lease(Control& value) : owner(value) {
            std::unique_lock<std::mutex> guard(owner.mutex);
            owner.ready.wait(guard, [&] { return !owner.free_slots.empty(); });
            slot = owner.free_slots.back();
            owner.free_slots.pop_back();
        }
        Lease(Lease const&) = delete;
        ~Lease() {
            {
                std::lock_guard<std::mutex> guard(owner.mutex);
                owner.free_slots.push_back(slot); // capacity fixed at construction
            }
            owner.ready.notify_one();
        }
    };
};

extern "C" char const* uc_error() { return last_error.c_str(); }
extern "C" char const* uc_version() { return "2.23.0"; }
extern "C" Control* uc_open(char const* path, std::size_t slots,
                             std::size_t dimension, std::size_t count, int metric) {
    try {
        if (!path || !slots || slots > 64) throw std::invalid_argument("invalid path/slots");
        auto loaded = index_dense_t::make(path, false);
        if (!loaded) throw std::runtime_error(loaded.error.release());
        if (loaded.index.dimensions() != dimension || loaded.index.size() != count ||
            loaded.index.scalar_kind() != scalar_kind_t::f32_k ||
            static_cast<int>(loaded.index.metric().metric_kind()) != metric || loaded.index.multi())
            throw std::invalid_argument("index shape/metric/dtype mismatch");
        return new Control(std::move(loaded.index), slots);
    } catch (std::exception const& e) { last_error = e.what(); return nullptr; }
      catch (...) { last_error = "unknown open failure"; return nullptr; }
}
// Private caller must join clients before destruction; the Python owner enforces
// this with a lifecycle gate, separate from the native scratch pool.
extern "C" void uc_close(Control* p) { delete p; }
extern "C" char const* uc_isa(Control const* p) { return p->index.metric().isa_name(); }
extern "C" int uc_configure(Control* p, std::size_t expansion) {
    try {
        if (!p || !expansion) throw std::invalid_argument("invalid expansion");
        std::unique_lock<std::shared_mutex> guard(p->configuration);
        p->index.change_expansion_search(expansion);
        return 0;
    } catch (std::exception const& e) { last_error = e.what(); return -1; }
      catch (...) { last_error = "unknown configure failure"; return -1; }
}
extern "C" int uc_search(Control* p, float const* query, std::size_t dimension,
                          std::size_t k, unsigned long long* keys, float* distances,
                          std::size_t* count) {
    try {
        if (!p || !query || !keys || !distances || !count || !k ||
            dimension != p->index.dimensions() || k > p->index.size())
            throw std::invalid_argument("invalid query/outputs/k");
        std::shared_lock<std::shared_mutex> guard(p->configuration);
        if (p->index.config().expansion_search < k) throw std::invalid_argument("expansion < k");
        Control::Lease lease(*p);
        auto result = p->index.search(query, k, lease.slot, false);
        if (!result) throw std::runtime_error(result.error.release());
        // Copy out while the lease is held: results borrow this slot's scratch.
        *count = result.dump_to(keys, distances, k);
        return 0;
    } catch (std::exception const& e) { last_error = e.what(); return -1; }
      catch (...) { last_error = "unknown search failure"; return -1; }
}

// Bounded, deterministic GIL-release witness for the private adapter tests.
// No query, index or timing path calls these functions.
struct Probe { std::mutex mutex; std::condition_variable cv; bool entered=false, released=false; };
extern "C" Probe* uc_probe_new() { return new Probe; }
extern "C" int uc_probe_wait(Probe* p) {
    std::unique_lock<std::mutex> lock(p->mutex);
    p->entered = true; p->cv.notify_all();
    return p->cv.wait_for(lock, std::chrono::seconds(2), [&] { return p->released; }) ? 0 : -1;
}
extern "C" int uc_probe_release(Probe* p) {
    std::unique_lock<std::mutex> lock(p->mutex);
    if (!p->cv.wait_for(lock, std::chrono::seconds(2), [&] { return p->entered; })) return -1;
    p->released = true; p->cv.notify_all(); return 0;
}
extern "C" void uc_probe_close(Probe* p) { delete p; }
