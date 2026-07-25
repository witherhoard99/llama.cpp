// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include <string>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <thread>

#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <unordered_set>
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if !defined(_WIN32)
// shrink region to page-interior only (round start up, end down)
#define EVICT_ALIGN(addr, size, page_size, out_addr, out_size) \
    do { \
        uintptr_t _s = ((uintptr_t)(addr) + (page_size) - 1) & ~(uintptr_t)(page_size - 1); \
        uintptr_t _e = ((uintptr_t)(addr) + (size)) & ~(uintptr_t)(page_size - 1); \
        if (_e <= _s) { \
            (out_addr) = NULL; \
            (out_size) = 0; \
        } else { \
            (out_addr) = (void *)_s; \
            (out_size) = _e - _s; \
        } \
    } while (0)
#endif

#define ENABLE_PREFETCH 0 // faster without N+1 prefetching. only ~1% hit rate. waste of resources

// qwen3-coder-next default
#define GGML_EXPERT_CACHE_MAX_DEFAULT 110
#define GGML_EXPERT_RAM_CACHE_MAX_DEFAULT 300
// 102 experts left live on disk. ~20% of experts

static int32_t g_expert_cache_max = GGML_EXPERT_CACHE_MAX_DEFAULT;
static int32_t g_expert_ram_cache_max = GGML_EXPERT_RAM_CACHE_MAX_DEFAULT;

static void resolve_expert_cache_config() {
    static bool resolved = false;
    if (resolved) return;
    resolved = true;
    const char * env_cache = getenv("GGML_EXPERT_CACHE_MAX");
    if (env_cache) {
        int val = atoi(env_cache);
        if (val > 0) g_expert_cache_max = val;
    }
    const char * env_ram = getenv("GGML_EXPERT_RAM_CACHE_MAX");
    if (env_ram) {
        int val = atoi(env_ram);
        if (val > 0) g_expert_ram_cache_max = val;
    }
}

#define GGML_EXPERT_CACHE
#define GGML_EXPERT_RAM_CACHE

// example usage:
// GGML_EXPERT_CACHE_MAX=90 GGML_EXPERT_RAM_CACHE_MAX=166 GGML_OP_OFFLOAD_MIN_BATCH=1 ./build/bin/llama-cli -fa on -ctk q8_0 -ctv q8_0 -m ~/Qwen3.5-35B-A3B-UD-IQ4_NL.gguf -c 32000 --jinja --mmap -ngl 99 --cpu-moe -t 11 -n 1000 -f test-prompt.txt --perf --single-turn
// GGML_EXPERT_CACHE_MAX=90 GGML_EXPERT_RAM_CACHE_MAX=166 GGML_OP_OFFLOAD_MIN_BATCH=1 ./build/bin/llama-server -fa on -ctk q8_0 -ctv q8_0 -m ~/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -c 256000 --jinja --mmap -ngl 99 --cpu-moe -t 11 -fit off -lv 4

#define GGML_EXPERT_CACHE_STRATEGY_LFU 1
#define GGML_EXPERT_CACHE_STRATEGY_LRU 2 // broken
#define GGML_EXPERT_CACHE_STRATEGY_LFU_AGING 8
#define GGML_EXPERT_CACHE_STRATEGY_LFU_AGING_GUARDED 13

#define GGML_EXPERT_CACHE_STRATEGY GGML_EXPERT_CACHE_STRATEGY_LFU_AGING // best
//#define GGML_EXPERT_CACHE_STRATEGY GGML_EXPERT_CACHE_STRATEGY_LFU

//#define GGML_EXPERT_AGING_THRESHOLD 16
#define GGML_EXPERT_AGING_THRESHOLD 128 // no real diff between 64 and 128 it seems
#define GGML_EXPERT_WINDOW_SIZE 256


#ifdef GGML_EXPERT_CACHE

#ifdef GGML_EXPERT_RAM_CACHE
typedef bool (*ggml_backend_register_host_buffer_fn)(void * buffer, size_t size);
typedef void (*ggml_backend_unregister_host_buffer_fn)(void * buffer);

static ggml_backend_register_host_buffer_fn   g_register_host_buffer   = nullptr;
static ggml_backend_unregister_host_buffer_fn g_unregister_host_buffer = nullptr;
static bool g_host_buf_fns_resolved = false;

static void resolve_host_buf_fns(ggml_backend_t backend) {
    if (g_host_buf_fns_resolved) return;
    g_host_buf_fns_resolved = true;
    if (!backend) return;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) return;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return;
    g_register_host_buffer = (ggml_backend_register_host_buffer_fn)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer");
    if (g_register_host_buffer) {
        g_unregister_host_buffer = (ggml_backend_unregister_host_buffer_fn)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_unregister_host_buffer");
    }
}

static struct io_uring g_ring = {};
static int g_ring_initialized = 0;
static uint32_t g_sector_size = 512;

#define DISK_XFER_SLOTS 32
#define DISK_XFER_N1_PREFETCH_SLOTS 4
#define DISK_XFER_N1_PREFETCH_LAYERS 8
#define DISK_XFER_N1_PREFETCH_DEPTH 4
#define DISK_XFER_N1_PREFETCH_TOTAL_SLOTS (DISK_XFER_N1_PREFETCH_SLOTS * DISK_XFER_N1_PREFETCH_LAYERS)
#define DISK_XFER_TOTAL_SLOTS (DISK_XFER_SLOTS + (DISK_XFER_N1_PREFETCH_SLOTS * DISK_XFER_N1_PREFETCH_LAYERS))
#define DISK_XFER_N1_PREFETCH_MIN_USAGE 1

static void *g_gate_xfer_buf = NULL;
static void *g_up_xfer_buf = NULL;
static void *g_down_xfer_buf = NULL;

// XXX: ffn_gate_up_exps will be classified as UP
enum CacheTensorType {
    TTYPE_GATE = 0,
    TTYPE_UP,
    TTYPE_DOWN,
};
#endif // GGML_EXPERT_RAM_CACHE

static int g_last_layer_id = INT_MAX;

struct disk_read {
    off_t aligned_offset;
    size_t aligned_len;
    size_t head_skip;
    void *slot;
    uint32_t exp_id;
    uint32_t run_length;
    struct ggml_expert_tensor_cache *cache;
    bool is_n1_prefetch;
};

struct ggml_expert_tensor_cache {
    std::string tensor_name;
    int layer_id;

    size_t expert_size;
    size_t n_experts; // total number of experts in this tensor
    size_t max_cached; // max number of cached experts supported for this tensor

    ggml_backend_buffer_t cache_buffer;
    void* d_cache_buffer;

    ggml_backend_buffer_t orig_input_cpy_buffer;
    void *orig_input_cpy_d_buffer;

    int32_t* cached_expert_ids; // the expert ids of the entries in the cache at respective position. mapping of cache id -> expert id
    int32_t* expert_id_of_cached; // mapping from expert id -> cache id. -1 if not in cache
    uint64_t* usage_counts;
    uint64_t max_usage_count;
    uint64_t* slot_access;

    int32_t n_cached; // number of cached experts
    bool initialized;

#ifdef GGML_EXPERT_RAM_CACHE
    // RAM cache (pinned staging buffer)
    void * pinned_staging;                          // page-aligned, cudaHostRegister'd buffer
    int32_t * staging_slot_of_expert_id;             // expert_id → staging slot (-1 if not in staging)
    int32_t * staging_expert_of_slot;                // staging slot → expert_id (-1 if free)
    std::vector<int32_t> free_staging_slots;        // available slot indices
    int32_t n_staging_slots;                         // total number of staging slots = max_staged
    int32_t max_staged;                              // = GGML_EXPERT_RAM_CACHE_MAX
    bool staging_registered;                         // whether cudaHostRegister succeeded

    int xfer_fd;                                    // O_DIRECT fd of the GGUF file where the weights for this tensor is stored. assumes all weights/experts for a given layer's tensors all go in the same file.
    off_t xfer_file_offset;                          // byte offset in file where input->data starts
    CacheTensorType tensor_type;                    // GATE, UP or DOWN
    ggml_expert_tensor_cache *next_cache;           // pointer to the next layers cache
                                                    //
    ggml_expert_tensor_cache *gate_cache;           // pointer to the gate cache for this layer
    ggml_expert_tensor_cache *up_cache;           // pointer to the up cache for this layer
    ggml_expert_tensor_cache *down_cache;           // pointer to the down cache for this layer

    bool disk_weights_copied;                                      // true after copy_expert_from_disk has run for this cache in the current decode step
    bool has_n1_prefetch;                                          // true after N+1 layer prefetch has been submitted for this cache

    std::vector<disk_read*> disk_inflight_experts;            // experts that have been submitted to io_uring but not finished yet
    std::vector<disk_read*> disk_ready_experts;              // experts that have been submitted to io_uring AND have finished and are sitting in the xfer buffer
#endif // GGML_EXPERT_RAM_CACHE
};

#ifdef GGML_EXPERT_RAM_CACHE
static void submit_disk_read(ggml_expert_tensor_cache *cache, uint32_t exp_id, uint32_t run_length, void *dst, bool is_n1_prefetch = false) {
    off_t raw_offset = cache->xfer_file_offset + (off_t)exp_id * cache->expert_size;
    off_t raw_end = raw_offset + (off_t)run_length * cache->expert_size;
    off_t aligned_offset = (raw_offset / (off_t)g_sector_size) * (off_t)g_sector_size;
    off_t aligned_end = ((raw_end + (off_t)g_sector_size - 1) / (off_t)g_sector_size) * (off_t)g_sector_size;

    struct disk_read *dr = new disk_read{};
    dr->aligned_offset = aligned_offset;
    dr->aligned_len = aligned_end - aligned_offset;
    dr->head_skip = raw_offset - aligned_offset;
    dr->slot = dst;
    dr->exp_id = exp_id;
    dr->run_length = run_length;
    dr->cache = cache;
    dr->is_n1_prefetch = is_n1_prefetch;
    cache->disk_inflight_experts.push_back(dr);

    struct io_uring_sqe *sqe = io_uring_get_sqe(&g_ring);
    if (!sqe) {
        io_uring_submit(&g_ring);
        sqe = io_uring_get_sqe(&g_ring);
    }
    GGML_ASSERT(sqe);

    io_uring_prep_read(sqe, cache->xfer_fd, dr->slot, dr->aligned_len, dr->aligned_offset);
    sqe->user_data = (uint64_t)(uintptr_t)dr;
}
#endif // GGML_EXPERT_RAM_CACHE

static std::vector<std::map<std::string, ggml_expert_tensor_cache>> g_expert_caches;
static uint64_t g_cache_hits;
static uint64_t g_cache_misses;
static uint64_t g_ram_hits;
static uint64_t g_disk_misses;
static uint64_t g_disk_wait_ns;
static uint64_t g_n1_prefetch_total;
static uint64_t g_n1_prefetch_hits;
uint64_t g_access_counter = 0;
uint64_t g_expert_cache_size = 0;
uint64_t g_expert_ram_cache_size = 0;
static std::atomic<bool> g_stats_thread_running{false};
static std::thread g_stats_thread;

static void ggml_expert_cache_stats_loop() {
    while (g_stats_thread_running.load()) {
        //std::this_thread::sleep_for(std::chrono::seconds(60));
        std::this_thread::sleep_for(std::chrono::seconds(10));
        if (!g_stats_thread_running.load()) break;
        uint64_t hits = g_cache_hits;
        uint64_t misses = g_cache_misses;
        uint64_t ram_hits = g_ram_hits;
        uint64_t disk_misses = g_disk_misses;
        uint64_t total = hits + ram_hits + disk_misses;
        double gpu_rate = total > 0 ? 100.0 * hits / total : 0.0;
        double ram_rate = total > 0 ? 100.0 * ram_hits / total : 0.0;
        double disk_rate = total > 0 ? 100.0 * disk_misses / total : 0.0;
        fprintf(stdout, "[expert cache] GPU=%llu (%.1f%%) RAM=%llu (%.1f%%) disk=%llu (%.1f%%) disk_wait=%.3fs\n",
            (unsigned long long)hits, gpu_rate,
            (unsigned long long)ram_hits, ram_rate,
            (unsigned long long)disk_misses, disk_rate,
            (double)g_disk_wait_ns / 1e9);
        uint64_t n1_total = g_n1_prefetch_total;
        uint64_t n1_hits = g_n1_prefetch_hits;
        if (n1_total > 0) {
            double n1_rate = 100.0 * n1_hits / n1_total;
            fprintf(stdout, "[expert cache] N+1 prefetch: %llu/%llu (%.1f%%)\n",
                (unsigned long long)n1_hits, (unsigned long long)n1_total, n1_rate);
        }
        fflush(stdout);
    }
}

static void ggml_expert_cache_stats_start() {
    if (!g_stats_thread_running.exchange(true)) {
        g_stats_thread = std::thread(ggml_expert_cache_stats_loop);
    }
}

static void ggml_expert_cache_stats_stop() {
    if (g_stats_thread_running.exchange(false)) {
        if (g_stats_thread.joinable()) {
            g_stats_thread.join();
        }
    }
}


static void ggml_expert_cache_free_internal();

static void ggml_expert_cache_atexit() {
    ggml_expert_cache_free_internal();
}

static void ggml_expert_cache_register_atexit() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        std::atexit(ggml_expert_cache_atexit);
    });
}

#ifdef GGML_EXPERT_RAM_CACHE
static void fd_from_mmap_ptr(ggml_expert_tensor_cache& cache, void *target_ptr)
{
    uintptr_t target = (uintptr_t)target_ptr;
    FILE *f = fopen("/proc/self/maps", "r");
    if (f) {
        char line[1024];
        while (fgets(line, sizeof(line), f)) {
            uintptr_t lo, hi;
            unsigned long offset;
            char perms[5];
            char path[512] = {0};
            int n = sscanf(line, "%lx-%lx %4s %lx %*d:%*d %*d %511[^\n]",
                    &lo, &hi, perms, &offset, path);
            if (n >= 4 && target >= lo && target < hi) {
                if (n >= 5 && path[0] != '\0') {
                    size_t plen = strlen(path);
                    while (plen > 0 && (path[plen-1] == ' ' || path[plen-1] == '\n' || path[plen-1] == '\r'))
                        path[--plen] = '\0';

                    cache.xfer_file_offset = (off_t)offset + (off_t)(target - lo);
                    int fd = open(path, O_RDONLY | O_DIRECT);
                    if (fd >= 0) {
                        cache.xfer_fd = fd;
                        //fprintf(stdout, "[expert cache] O_DIRECT fd for %s: %s (file_offset=%ld)\n",
                        //        cache.tensor_name.c_str(), path, (long)cache.xfer_file_offset);
                    } else {
                        fprintf(stderr, "failed opening GGUF file in O_DIRECT!\n");
                        abort();
                    }
                }
                break;
            }
        }
        fclose(f);
    }
    if (cache.xfer_fd < 0) {
        fprintf(stderr, "[expert cache] could not find backing file for %s in /proc/self/maps\n",
                cache.tensor_name.c_str());
    }
}
#endif

static void ggml_expert_cache_init(
    ggml_expert_tensor_cache& cache,
    int layer_id,
    const std::string& tensor_name,
    size_t n_experts,
    size_t expert_size,
    ggml_backend_t backend,
    struct ggml_tensor * input
) {
    resolve_expert_cache_config();

    if (cache.initialized) return;


    static ggml_expert_tensor_cache *prev_up_cache = NULL;
    static ggml_expert_tensor_cache *prev_down_cache = NULL;
    static ggml_expert_tensor_cache *prev_gate_cache = NULL;

    cache.layer_id = layer_id;
    cache.tensor_name = tensor_name;
    cache.n_experts = n_experts;
    cache.expert_size = expert_size;
    cache.max_cached = g_expert_cache_max;
    cache.n_cached = 0;

    size_t cache_buffer_size = cache.max_cached * cache.expert_size;
    cache.cache_buffer = ggml_backend_alloc_buffer(backend, cache_buffer_size);
    cache.cache_buffer->usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE; // same as input_cpy->buffer
    cache.d_cache_buffer = ggml_backend_buffer_get_base(cache.cache_buffer);
    g_expert_cache_size += cache_buffer_size;
    fprintf(stdout, "[expert cache] init: caching %ld experts from layer %s. GPU expert cache size: %ld MiB\n", cache.max_cached, tensor_name.c_str(), g_expert_cache_size / 1024 / 1024);

    cache.orig_input_cpy_buffer = NULL;
    cache.orig_input_cpy_d_buffer = NULL;

    cache.cached_expert_ids = (int32_t*) calloc(cache.max_cached, sizeof(int32_t));
    cache.expert_id_of_cached = (int32_t*) calloc(cache.n_experts, sizeof(int32_t));
    cache.usage_counts = (uint64_t*) calloc(cache.n_experts, sizeof(uint64_t));
    cache.max_usage_count = 0;
    cache.slot_access = (uint64_t*) calloc(cache.n_experts, sizeof(uint64_t));

    for (size_t i = 0; i < cache.max_cached; i++) {
        cache.cached_expert_ids[i] = -1;
    }
    for (size_t i = 0; i < cache.n_experts; i++) {
        cache.expert_id_of_cached[i] = -1;
    }


    cache.initialized = true;

#ifdef GGML_EXPERT_RAM_CACHE
    cache.n_staging_slots = g_expert_ram_cache_max;
    cache.max_staged = g_expert_ram_cache_max;
    cache.staging_registered = false;
    cache.pinned_staging = nullptr;
    long page_size = sysconf(_SC_PAGESIZE);

    g_expert_ram_cache_size += cache.n_staging_slots * cache.expert_size;
    fprintf(stdout, "[expert cache] RAM expert cache size: %ld MiB\n", g_expert_ram_cache_size / 1024 / 1024);

    setenv("GGML_CUDA_REGISTER_HOST", "1", 1);
    resolve_host_buf_fns(backend);

    if (cache.n_staging_slots > 0) {
        int ret = posix_memalign(&cache.pinned_staging, page_size,
                                  cache.n_staging_slots * expert_size);
        if (ret != 0 || cache.pinned_staging == nullptr) {
            fprintf(stderr, "%s: failed to allocate pinned staging buffer for %s: %s\n",
                    __func__, tensor_name.c_str(), strerror(ret));
            cache.pinned_staging = nullptr;
        } else if (g_register_host_buffer) {
            bool ok = g_register_host_buffer(cache.pinned_staging,
                    cache.n_staging_slots * expert_size);
            if (ok) {
                cache.staging_registered = true;
                fprintf(stdout, "[expert cache] registered staging buffer for %s (%zu MiB)\n",
                        tensor_name.c_str(), cache.n_staging_slots * expert_size / 1024 / 1024);
            } else {
                fprintf(stderr, "%s: cudaHostRegister failed for %s "
                        "(async copies will be synchronous)\n",
                        __func__, tensor_name.c_str());
            }
        } else {
            fprintf(stderr, "%s: cudaHostRegister not available for %s "
                    "(async copies will be synchronous)\n",
                    __func__, tensor_name.c_str());
        }
    }

    cache.staging_slot_of_expert_id = (int32_t*) calloc(cache.n_experts, sizeof(int32_t));
    cache.staging_expert_of_slot = (int32_t*) calloc(cache.n_staging_slots, sizeof(int32_t));

    for (size_t i = 0; i < cache.n_experts; i++) {
        cache.staging_slot_of_expert_id[i] = -1;
    }
    for (int32_t i = 0; i < cache.n_staging_slots; i++) {
        cache.staging_expert_of_slot[i] = -1;
    }

    // Pre-populate: put the first max_cached experts in GPU cache,
    // and the next max_staged experts in the RAM cache (staging buffer).
    if (cache.staging_registered && cache.n_staging_slots > 0 && cache.pinned_staging) {
        // Pre-populate GPU cache with first max_cached experts
        for (int32_t i = 0; i < (int32_t)cache.max_cached; i++) {
            cache.cached_expert_ids[i] = i;
            cache.expert_id_of_cached[i] = i;
        }
        cache.n_cached = cache.max_cached;

        // Bulk copy GPU cache entries from input
        {
            ggml_tensor cache_dst = {};
            cache_dst.data = cache.d_cache_buffer;
            cache_dst.buffer = cache.cache_buffer;
            cache_dst.ne[0] = input->ne[0];
            cache_dst.ne[1] = input->ne[1];
            cache_dst.ne[2] = cache.max_cached;
            cache_dst.ne[3] = 1;
            cache_dst.type = input->type;
            cache_dst.nb[0] = input->nb[0];
            cache_dst.nb[1] = input->nb[1];
            cache_dst.nb[2] = cache.expert_size;
            cache_dst.nb[3] = cache.max_cached * cache.expert_size;

            ggml_backend_tensor_set_async(backend, &cache_dst,
                input->data, 0, cache.max_cached * cache.expert_size);
            ggml_backend_synchronize(backend);
        }

        // Pre-populate RAM cache with remaining experts
        int32_t slot = 0;
        for (int32_t i = (int32_t)cache.max_cached; i < (int32_t)cache.n_experts; i++) {
            if (slot >= cache.n_staging_slots) break;

            cache.staging_slot_of_expert_id[i] = slot;
            cache.staging_expert_of_slot[slot] = i;

            memcpy((char*)cache.pinned_staging + slot * cache.expert_size,
                   (char*)input->data + i * cache.expert_size,
                   cache.expert_size);
            slot++;
        }

        for (; slot < cache.n_staging_slots; slot++) {
            cache.free_staging_slots.push_back(slot);
        }

#if !defined(_WIN32)
        {
            const long ps = sysconf(_SC_PAGESIZE);
            for (int32_t i = 0; i < (int32_t)cache.n_experts; i++) {
                void *addr; size_t sz;
                EVICT_ALIGN((char*)input->data + i * cache.expert_size,
                             cache.expert_size, ps, addr, sz);
                if (sz > 0) {
                    madvise(addr, sz, MADV_DONTNEED);
                }
            }
        }
#endif

        fprintf(stdout, "[expert cache] RAM cache pre-populated for %s (%d/%d experts in RAM)\n",
                tensor_name.c_str(), (int)cache.n_staging_slots - (int)cache.free_staging_slots.size(), cache.n_staging_slots);
    } else {
        for (int32_t i = 0; i < cache.n_staging_slots; i++) {
            cache.free_staging_slots.push_back(i);
        }
    }

    fd_from_mmap_ptr(cache, input->data);

    ggml_expert_tensor_cache **prev_cache = NULL;
    void **xfer_buf = NULL;

    if (cache.tensor_name.find("up_exps") != std::string::npos) {
        cache.tensor_type = TTYPE_UP;
        prev_cache = &prev_up_cache;
        xfer_buf = &g_up_xfer_buf;
    } else if (cache.tensor_name.find("down_exps") != std::string::npos) {
        cache.tensor_type = TTYPE_DOWN;
        prev_cache = &prev_down_cache;
        xfer_buf = &g_down_xfer_buf;
    } else if (cache.tensor_name.find("gate_exps") != std::string::npos) {
        cache.tensor_type = TTYPE_GATE;
        prev_cache = &prev_gate_cache;
        xfer_buf = &g_gate_xfer_buf;
    } else {
        fprintf(stderr, "[expert cache] UNHANDLED TENSOR TYPE '%s'\n", cache.tensor_name.c_str());
        abort();
    }

    if (*prev_cache) {
        (*prev_cache)->next_cache = &cache;
    }

    *prev_cache = &cache;

    if (!g_ring_initialized) {
        int sector_sz = 512;
        ioctl(cache.xfer_fd, BLKSSZGET, &sector_sz);
        g_sector_size = sector_sz;

        int ret = io_uring_queue_init(64, &g_ring, 0);
        if (ret == 0) {
            g_ring_initialized = true;
            fprintf(stdout, "[expert cache] io_uring ring initialized (depth=64)\n");
        } else {
            fprintf(stderr, "[expert cache] io_uring_queue_init failed: %s\n", strerror(-ret));
            abort();
        }
    }

    if (!(*xfer_buf)) {
        size_t slot_stride = ((cache.expert_size + g_sector_size*2 + g_sector_size - 1) / g_sector_size) * g_sector_size;

        int ret = posix_memalign(xfer_buf, page_size,
                                  slot_stride * DISK_XFER_TOTAL_SLOTS);

        if (ret != 0 || *xfer_buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate disk xfer buffer for %s: %s\n",
                    __func__, cache.tensor_name.c_str(), strerror(ret));
            abort();
        }

        bool ok = g_register_host_buffer && g_register_host_buffer(*xfer_buf,
                slot_stride * DISK_XFER_TOTAL_SLOTS);
        if (ok) {
            cache.staging_registered = true;
            fprintf(stdout, "[expert cache] registered disk staging buffer for type %d (%zu MiB)\n",
                    cache.tensor_type, (slot_stride * DISK_XFER_TOTAL_SLOTS) / 1024 / 1024);
        } else {
            fprintf(stderr, "%s: cudaHostRegister failed for %s "
                    "(async copies will be synchronous)\n",
                    __func__, tensor_name.c_str());
        }
    }

    // set self current layer cache pointers
    if (cache.tensor_type == TTYPE_GATE)   cache.gate_cache   = &cache;
    if (cache.tensor_type == TTYPE_UP)     cache.up_cache     = &cache;
    if (cache.tensor_type == TTYPE_DOWN)   cache.down_cache   = &cache;

    // set sibling current layer cache pointers
    auto& layer_map = g_expert_caches[layer_id];
    for (auto& [name, sibling] : layer_map) {
        if (!sibling.initialized) continue;
        if (sibling.tensor_type == TTYPE_GATE)   cache.gate_cache   = &sibling;
        if (sibling.tensor_type == TTYPE_UP)     cache.up_cache     = &sibling;
        if (sibling.tensor_type == TTYPE_DOWN)   cache.down_cache   = &sibling;

        if (&sibling != &cache) {
            if (cache.tensor_type == TTYPE_GATE)   sibling.gate_cache   = &cache;
            if (cache.tensor_type == TTYPE_UP)     sibling.up_cache     = &cache;
            if (cache.tensor_type == TTYPE_DOWN)   sibling.down_cache   = &cache;
        }
    }


#endif // GGML_EXPERT_RAM_CACHE (staging init)

#if 0
    // Pre-populate the first g_expert_cache_max experts into the cache
    // this results in worse generation t/s somehow
    for (size_t i = 0; i < (size_t)g_expert_cache_max; i++) {
        cache.cached_expert_ids[i] = i;
        cache.expert_id_of_cached[i] = i;
        //cache.usage_counts[i] = 0;
        //cache.slot_access[i] = g_access_counter++;
    }
    cache.n_cached = g_expert_cache_max;

    // Single bulk copy of all pre-populated experts from the input tensor to the cache buffer
    ggml_tensor cache_dst = {};
    cache_dst.data   = cache.d_cache_buffer;
    cache_dst.buffer = cache.cache_buffer;
    cache_dst.ne[0]  = input->ne[0];
    cache_dst.ne[1]  = input->ne[1];
    cache_dst.ne[2]  = g_expert_cache_max;
    cache_dst.ne[3]  = 1;
    cache_dst.type   = input->type;
    cache_dst.nb[0]  = input->nb[0];
    cache_dst.nb[1]  = input->nb[1];
    cache_dst.nb[2]  = expert_size;
    cache_dst.nb[3]  = g_expert_cache_max * expert_size;

    ggml_backend_tensor_set_async(backend,
            &cache_dst,
            input->data,
            0,
            g_expert_cache_max * expert_size);
    ggml_backend_synchronize(backend);

    //fprintf(stdout, "[expert cache] pre-populated: layer=%d, tensor=%s, n_experts=%zu\n",
            //layer_id, tensor_name.c_str(), n_experts);
#endif

    ggml_expert_cache_register_atexit();
    ggml_expert_cache_stats_start();

    GGML_LOG_DEBUG("Expert cache initialized: layer=%d, tensor=%s, n_experts=%zu, expert_size=%zu\n",
        layer_id, tensor_name.c_str(), n_experts, expert_size);
}

static void ggml_expert_cache_print_stats(void) {
    FILE* f = fopen("./llama_expert_usage.txt", "w");
    if (!f) return;

    fprintf(f, "\n=== Expert GPU Cache Statistics ===\n");
    fprintf(f, "Cache hits:   %llu\n", (unsigned long long)g_cache_hits);
    fprintf(f, "Cache misses: %llu\n", (unsigned long long)g_cache_misses);
    if (g_cache_hits + g_cache_misses > 0) {
        double hit_rate = 100.0 * g_cache_hits / (g_cache_hits + g_cache_misses);
        fprintf(f, "Hit rate:     %.2f%%\n", hit_rate);
    }

    for (size_t layer = 0; layer < g_expert_caches.size(); layer++) {
        for (auto& [tensor_name, cache] : g_expert_caches[layer]) {
            if (cache.initialized) {
                size_t actual_cached = 0;
                for (size_t i = 0; i < cache.max_cached; i++) {
                    if (cache.cached_expert_ids[i] >= 0) actual_cached++;
                }
                fprintf(f, "\nLayer %zu, Tensor %s:\n", layer, tensor_name.c_str());
                fprintf(f, "  Cached experts: %zu/%zu\n", actual_cached, cache.max_cached);
                for (size_t i = 0; i < cache.max_cached; i++) {
                    if (cache.cached_expert_ids[i] >= 0) {
                        int32_t exp_id = cache.cached_expert_ids[i];
                        fprintf(f, "  Slot %zu: expert %d (cached), usage=%llu, last_access=%llu\n", i, exp_id, (unsigned long long)cache.usage_counts[exp_id], (unsigned long long)cache.slot_access[exp_id]);
                    }
                }
                for (size_t e = 0; e < cache.n_experts; e++) {
                    bool in_cache = false;
                    for (size_t i = 0; i < cache.max_cached; i++) {
                        if (cache.cached_expert_ids[i] == (int32_t)e) {
                            in_cache = true;
                            break;
                        }
                    }
                    if (!in_cache && cache.usage_counts[e] > 0) {
                        fprintf(f, "  Expert %zu (not cached), usage=%llu, last_access=%lu\n", e, (unsigned long long)cache.usage_counts[e], cache.slot_access[e]);
                    }
                }
            }
        }
    }

    fprintf(f, "=====================================\n");
    fclose(f);
}

static void ggml_expert_cache_free_internal() {
    ggml_expert_cache_stats_stop();
    ggml_expert_cache_print_stats();
    for (auto& layer_map : g_expert_caches) {
        for (auto& [name, cache] : layer_map) {
            if (cache.initialized) {
#ifdef GGML_EXPERT_RAM_CACHE
                if (cache.staging_registered && g_unregister_host_buffer) {
                    g_unregister_host_buffer(cache.pinned_staging);
                    cache.staging_registered = false;
                }
                if (cache.pinned_staging) {
                    free(cache.pinned_staging);
                    cache.pinned_staging = nullptr;
                }
                free(cache.staging_slot_of_expert_id);
                free(cache.staging_expert_of_slot);
                cache.free_staging_slots.clear();
#endif // GGML_EXPERT_RAM_CACHE

                ggml_backend_buffer_free(cache.cache_buffer);
                free(cache.cached_expert_ids);
                free(cache.expert_id_of_cached);
                free(cache.usage_counts);
                free(cache.slot_access);
            }
        }
    }
    g_expert_caches.clear();
}

#endif // GGML_EXPERT_CACHE

// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}


#ifdef GGML_EXPERT_CACHE
#ifdef GGML_EXPERT_RAM_CACHE

static void wait_for_reads_finish(ggml_expert_tensor_cache *cache,
                                  ggml_backend_t split_backend,
                                  struct ggml_tensor *input_cpy,
                                  const std::vector<uint32_t> *active_exp_ids)
{
    auto t0 = std::chrono::steady_clock::now();
    size_t expert_size = cache->expert_size;

    auto is_active = [&](uint32_t exp_id) -> bool {
        if (!active_exp_ids) return true;
        for (auto id : *active_exp_ids) { if (id == exp_id) return true; }
        return false;
    };

    auto count_active_inflight = [&]() -> size_t {
        size_t count = 0;
        for (auto *dr : cache->disk_inflight_experts) {
            if (is_active(dr->exp_id)) count++;
        }
        return count;
    };

    // first, copy any already-completed reads to GPU before waiting for in-flight ones
    //fprintf(stderr, "[expert cache] wait_for_reads_finish: %s layer=%d ready=%zu inflight=%zu active_ids=%zu\n",
        //cache->tensor_name.c_str(), cache->layer_id, cache->disk_ready_experts.size(), cache->disk_inflight_experts.size(), active_exp_ids ? active_exp_ids->size() : 0);
    for (auto *dr : cache->disk_ready_experts) {
        //GGML_ASSERT(dr->cache == cache);
        //fprintf(stderr, "[expert cache]   ready_entry: exp_id=%u run_length=%u slot=%p is_active=%d\n",
            //dr->exp_id, dr->run_length, dr->slot, active_exp_ids ? (int)is_active(dr->exp_id) : -1);
        if (is_active(dr->exp_id)) {
            if (dr->is_n1_prefetch) g_n1_prefetch_hits++;
            void *expert_data = (char *)dr->slot + dr->head_skip;
            size_t dst_offset = (size_t)dr->exp_id * expert_size;
            //fprintf(stderr, "[expert cache]  %s: disk ready copying exp_id=%u run_length=%u slot=%p to GPU\n", dr->cache->tensor_name.c_str(), dr->exp_id, dr->run_length, dr->slot);
            ggml_backend_tensor_set_async(split_backend, input_cpy, expert_data, dst_offset, (size_t)dr->run_length * expert_size);
        }
        delete dr;
    }
    cache->disk_ready_experts.clear();

    while (cache->disk_inflight_experts.size() > 0) {
        struct io_uring_cqe *cqe;
        //io_uring_wait_cqe(&g_ring, &cqe);
        if (count_active_inflight() > 0) {
            io_uring_wait_cqe(&g_ring, &cqe);
        } else {
            if (io_uring_peek_cqe(&g_ring, &cqe) != 0) break;
        }
        if (cqe->res < 0) {
            fprintf(stderr, "[expert cache] io_uring read failed: %s\n", strerror(-cqe->res));
            abort();
        }

        struct disk_read *dr = (struct disk_read *)(uintptr_t)cqe->user_data;
        io_uring_cqe_seen(&g_ring, cqe);

        if (dr->cache == nullptr) {
            //fprintf(stderr, "[expert cache] ignoring slot=%p exp_id=%u with nullptr cache.\n", dr->slot, dr->exp_id);
            delete dr;
            continue;
        }

        ggml_expert_tensor_cache *owner = dr->cache;
        auto it = std::find(owner->disk_inflight_experts.begin(), owner->disk_inflight_experts.end(), dr);
        if (it != owner->disk_inflight_experts.end()) {
            owner->disk_inflight_experts.erase(it);
        }

        if (owner == cache) {
            if (is_active(dr->exp_id)) {
                if (dr->is_n1_prefetch) g_n1_prefetch_hits++;
                void *expert_data = (char *)dr->slot + dr->head_skip;
                size_t dst_offset = (size_t)dr->exp_id * expert_size;
                //fprintf(stderr, "[expert cache]  %s inflight finish copying exp_id=%u run_length=%u to GPU\n", dr->cache->tensor_name.c_str(), dr->exp_id, dr->run_length);
                ggml_backend_tensor_set_async(split_backend, input_cpy, expert_data, dst_offset, (size_t)dr->run_length * expert_size);
            }
            delete dr;
        } else {
            //fprintf(stderr, "[expert cache]   CQE for other owner: %s layer=%d exp_id=%u slot=%p\n",
                //owner->tensor_name.c_str(), owner->layer_id, dr->exp_id, dr->slot);
            owner->disk_ready_experts.push_back(dr);
        }
    }
    //ggml_backend_synchronize(split_backend);
    auto t1 = std::chrono::steady_clock::now();
    g_disk_wait_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

static void *get_xfer_buf(ggml_expert_tensor_cache *cache)
{
    if (cache->tensor_type == TTYPE_GATE) return g_gate_xfer_buf;
    if (cache->tensor_type == TTYPE_UP)   return g_up_xfer_buf;
    if (cache->tensor_type == TTYPE_DOWN) return g_down_xfer_buf;
    return NULL;
}

static std::vector<uint32_t> get_top_n_disk_experts(ggml_expert_tensor_cache *cache, int n) {
    std::vector<std::pair<uint64_t, uint32_t>> candidates;
    for (int32_t i = 0; i < cache->n_experts; i++) {
        if (cache->expert_id_of_cached[i] == -1 && cache->staging_slot_of_expert_id[i] == -1 && cache->usage_counts[i] >= DISK_XFER_N1_PREFETCH_MIN_USAGE) {
            candidates.push_back({cache->usage_counts[i], (uint32_t)i});
        }
    }
    std::sort(candidates.rbegin(), candidates.rend());
    std::vector<uint32_t> result;
    for (int i = 0; i < n && i < (int)candidates.size(); i++) {
        result.push_back(candidates[i].second);
    }
    return result;
}

static size_t count_occupied_main_slots(ggml_expert_tensor_cache *cache, void *buf, size_t slot_stride) {
    size_t occupied = 0;
    void *main_end = (char *)buf + DISK_XFER_SLOTS * slot_stride;
    for (auto *dr : cache->disk_inflight_experts) {
        if (dr->slot >= buf && dr->slot < main_end) occupied += dr->run_length;
    }
    for (auto *dr : cache->disk_ready_experts) {
        if (dr->slot >= buf && dr->slot < main_end) occupied += dr->run_length;
    }
    return occupied;
}

// copy expert from disk using io_uring instead of mmap, with sibling prefetch
// TODO: add a list of already opened files in the cache fd init part, so we don't have tons of fd's to the same file
static void copy_expert_from_disk(ggml_backend_t split_backend, struct ggml_tensor *input_cpy, ggml_expert_tensor_cache *cache, std::vector<uint32_t> *exp_ids, bool wait = true)
{
    size_t expert_size = cache->expert_size;
    size_t n = exp_ids->size();
    size_t sector_size = g_sector_size;
    // each slot must be sector-aligned, so round up the stride to a multiple of sector_size
    size_t slot_stride = ((expert_size + sector_size * 2 + sector_size - 1) / sector_size) * sector_size;
    void *buf = get_xfer_buf(cache);

    // build lookup of experts already fetched or in-flight from a sibling's prefetch
    std::unordered_set<uint32_t> exp_ids_set(exp_ids->begin(), exp_ids->end());
    std::unordered_set<uint32_t> already_fetched;
    for (auto *dr : cache->disk_ready_experts) {
        for (uint32_t r = 0; r < dr->run_length; r++) {
            uint32_t eid = dr->exp_id + r;
            if (exp_ids_set.count(eid)) already_fetched.insert(eid);
        }
    }
    for (auto *dr : cache->disk_inflight_experts) {
        for (uint32_t r = 0; r < dr->run_length; r++) {
            uint32_t eid = dr->exp_id + r;
            if (exp_ids_set.count(eid)) already_fetched.insert(eid);
        }
    }

    // copy any already-completed prefetched experts to GPU immediately
    {
        auto is_active = [&](uint32_t exp_id) -> bool {
            for (auto id : *exp_ids) { if (id == exp_id) return true; }
            return false;
        };
        for (auto *dr : cache->disk_ready_experts) {
            if (is_active(dr->exp_id)) {
                if (dr->is_n1_prefetch) g_n1_prefetch_hits++;
                void *expert_data = (char *)dr->slot + dr->head_skip;
                size_t dst_offset = (size_t)dr->exp_id * expert_size;
                //fprintf(stderr, "[expert cache] %s prefetched disk ready copying exp_id=%u run_length=%u to GPU\n", dr->cache->tensor_name.c_str(), dr->exp_id, dr->run_length);
                ggml_backend_tensor_set_async(split_backend, input_cpy, expert_data, dst_offset, (size_t)dr->run_length * expert_size);
            }
            delete dr;
        }
        // if we're gonna be reading entries into the slots we're copying over to the GPU then we need to sync before writing to them.
        if (cache->disk_ready_experts.size() > 0) {
            ggml_backend_synchronize(split_backend);
        }
        cache->disk_ready_experts.clear();
    }


    // count xfer buffer slots already occupied by prefetched data in the main area
    size_t occupied_slots = count_occupied_main_slots(cache, buf, slot_stride);

    //fprintf(stderr, "[expert cache] %s layer=%d copy_expert_from_disk: n=%zu already_fetched=%zu occupied_slots=%zu exp_ids=[",
        //cache->tensor_name.c_str(), cache->layer_id, n, already_fetched.size(), occupied_slots);
    //for (size_t ei = 0; ei < n && ei < 10; ei++) fprintf(stderr, "%s%u", ei ? "," : "", (*exp_ids)[ei]);
    //if (n > 10) fprintf(stderr, "...");
    //fprintf(stderr, "]\n");

    for (size_t base = 0; base < n; base += DISK_XFER_SLOTS) {
        size_t batch = std::min(n - base, (size_t)DISK_XFER_SLOTS);
        size_t slot_idx = occupied_slots;

        // track xfer buffer slot offsets per sibling across all experts in this batch
        ggml_expert_tensor_cache *siblings[] = { cache->gate_cache, cache->up_cache, cache->down_cache };
        size_t sib_slot_idx[] = { 0, 0, 0 };
        for (int si = 0; si < 3; si++) {
            size_t sib_occupied = 0;
            if (siblings[si]) {
                sib_occupied = count_occupied_main_slots(siblings[si], get_xfer_buf(siblings[si]),
                    ((siblings[si]->expert_size + sector_size * 2 + sector_size - 1) / sector_size) * sector_size);
            }
            sib_slot_idx[si] = sib_occupied;
        }

        for (size_t i = 0; i < batch; ) {
            uint32_t exp_id = (*exp_ids)[base + i];

            if (already_fetched.count(exp_id)) {
                //fprintf(stderr, "[expert cache] %s layer=%d SKIP exp_id=%u (already_fetched, inflight=%zu ready=%zu)\n",
                    //cache->tensor_name.c_str(), cache->layer_id, exp_id,
                    //cache->disk_inflight_experts.size(), cache->disk_ready_experts.size());
                i++; continue;
            }

            // detect consecutive experts for a merged read
            uint32_t run_length = 1;
            while (i + run_length < batch &&
                   (*exp_ids)[base + i + run_length] == exp_id + run_length &&
                   !already_fetched.count(exp_id + run_length)) {
                run_length++;
            }

            // if xfer buffer is full, reap completed reads to free slots
            if (slot_idx + run_length > (size_t)DISK_XFER_SLOTS) {
                io_uring_submit(&g_ring);
                wait_for_reads_finish(cache, split_backend, input_cpy, exp_ids);
                ggml_backend_synchronize(split_backend);
                slot_idx = 0;
                occupied_slots = 0;
                for (int si = 0; si < 3; si++) {
                    size_t sib_occupied = 0;
                    if (siblings[si]) {
                        sib_occupied = count_occupied_main_slots(siblings[si], get_xfer_buf(siblings[si]),
                            ((siblings[si]->expert_size + sector_size * 2 + sector_size - 1) / sector_size) * sector_size);
                    }
                    sib_slot_idx[si] = sib_occupied;
                }
            }

            // make sure the run fits in remaining xfer buffer slots
            if (slot_idx + run_length > (size_t)DISK_XFER_SLOTS) {
                run_length = (size_t)DISK_XFER_SLOTS - slot_idx;
            }

            submit_disk_read(cache, exp_id, run_length, (char *)buf + slot_idx * slot_stride);

            //fprintf(stderr, "[expert cache] %s layer=%d READ exp_id=%u run_length=%u\n", cache->tensor_name.c_str(), cache->layer_id, exp_id, run_length);

            // prefetch same experts for sibling caches (up/down/gate) so their reads overlap with current computation
            for (int si = 0; si < 3; si++) {
                auto *sib = siblings[si];
                if (!sib || sib == cache) continue;

                if (sib->disk_weights_copied) continue;

                // check if sibling already has any expert in this run or if its xfer buffer is full
                bool sib_has_any = false;
                for (uint32_t r = 0; r < run_length && !sib_has_any; r++) {
                    for (auto *sib_dr : sib->disk_inflight_experts) { if (sib_dr->exp_id == exp_id + r) { sib_has_any = true; break; } }
                    if (!sib_has_any) { for (auto *sib_dr : sib->disk_ready_experts) { if (sib_dr->exp_id == exp_id + r) { sib_has_any = true; break; } } }
                }
                if (sib_has_any) continue;
                if (sib_slot_idx[si] + run_length > (size_t)DISK_XFER_SLOTS) continue;

                void *sib_buf = get_xfer_buf(sib);
                if (!sib_buf) continue;

                size_t sib_slot_stride = ((sib->expert_size + sector_size * 2 + sector_size - 1) / sector_size) * sector_size;

                submit_disk_read(sib, exp_id, run_length, (char *)sib_buf + sib_slot_idx[si] * sib_slot_stride);

                //fprintf(stderr, "[expert cache] %s layer=%d PREFETCH exp_id=%u run_length=%u\n", sib->tensor_name.c_str(), sib->layer_id, exp_id, run_length);

                sib_slot_idx[si] += run_length;
            }

            slot_idx += run_length;
            i += run_length;
        }

        io_uring_submit(&g_ring);

        // reap CQEs and copy to GPU immediately, overlapping PCI-e with remaining disk I/O
        if (wait || base + DISK_XFER_SLOTS < n) {
            wait_for_reads_finish(cache, split_backend, input_cpy, exp_ids);
            ggml_backend_synchronize(split_backend); // if we're gonna be reading more entries into the slots we're copying over to the GPU then we need to sync
        }
    }

    cache->disk_weights_copied = true;

#if ENABLE_PREFETCH
    // prefetch top-N most used disk experts for upcoming layers (N+1 .. N+DEPTH)
    {
        bool submitted = false;
        ggml_expert_tensor_cache *depth_cache = cache;
        for (int depth = 0; depth < DISK_XFER_N1_PREFETCH_DEPTH && depth_cache != nullptr; depth++) {
            depth_cache = depth_cache->next_cache;
            if (!depth_cache) break;

            ggml_expert_tensor_cache *n1_caches[] = {
                depth_cache->gate_cache,
                depth_cache->up_cache,
                depth_cache->down_cache,
            };

            for (int si = 0; si < 3; si++) {
                ggml_expert_tensor_cache *sib = n1_caches[si];
                if (!sib || sib->has_n1_prefetch) continue;

                void *sib_buf = get_xfer_buf(sib);
                size_t sib_slot_stride = ((sib->expert_size + sector_size * 2 + sector_size - 1) / sector_size) * sector_size;

                // find the first available slot region for N+1 prefetch
                size_t n1_slot_offset = 0;
                for (int li = 0; li < DISK_XFER_N1_PREFETCH_LAYERS; li++) {
                    size_t region_start = DISK_XFER_SLOTS + li * DISK_XFER_N1_PREFETCH_SLOTS;
                    void *n1_start = (char *)sib_buf + region_start * sib_slot_stride;
                    void *n1_end = (char *)sib_buf + (region_start + DISK_XFER_N1_PREFETCH_SLOTS) * sib_slot_stride;
                    bool region_busy = false;
                    // find layer 0 cache of the same type, then walk next_cache chain
                    for (auto& [name, c] : g_expert_caches[0]) {
                        if (c.tensor_type == sib->tensor_type) {
                            for (ggml_expert_tensor_cache *p = &c; p != nullptr && !region_busy; p = p->next_cache) {
                                for (auto *dr : p->disk_inflight_experts) {
                                    if (dr->slot >= n1_start && dr->slot < n1_end) {
                                        fprintf(stderr, "[expert cache] %s layer=%d N+1 prefetch region %d busy (inflight %s layer=%d exp_id=%u)\n",
                                            sib->tensor_name.c_str(), sib->layer_id, li, p->tensor_name.c_str(), p->layer_id, dr->exp_id);
                                        region_busy = true; break;
                                    }
                                }
                                for (auto *dr : p->disk_ready_experts) {
                                    if (dr->slot >= n1_start && dr->slot < n1_end) {
                                        fprintf(stderr, "[expert cache] %s layer=%d N+1 prefetch region %d busy (ready %s layer=%d exp_id=%u)\n",
                                            sib->tensor_name.c_str(), sib->layer_id, li, p->tensor_name.c_str(), p->layer_id, dr->exp_id);
                                        region_busy = true; break;
                                    }
                                }
                            }
                            break;
                        }
                    }
                    if (!region_busy) {
                        n1_slot_offset = region_start;
                        break;
                    }
                }
                if (n1_slot_offset == 0) continue;

                sib->has_n1_prefetch = true;

                auto top_experts = get_top_n_disk_experts(sib, DISK_XFER_N1_PREFETCH_SLOTS);
                if (top_experts.empty()) continue;

                submitted = true;

                for (size_t ei = 0; ei < top_experts.size(); ei++) {
                    uint32_t exp_id = top_experts[ei];
                    submit_disk_read(sib, exp_id, 1, (char *)sib_buf + (n1_slot_offset + ei) * sib_slot_stride, true);
                    g_n1_prefetch_total++;

                    fprintf(stderr, "[expert cache] %s from_layer=%d to_layer=%d N+%d PREFETCH exp_id=%u slot=%zu\n", sib->tensor_name.c_str(), cache->layer_id, sib->layer_id, depth + 1, exp_id, n1_slot_offset + ei);
                }
            }
        }

        if (submitted) io_uring_submit(&g_ring);
    }
#endif
}

#endif // GGML_EXPERT_RAM_CACHE
#endif // GGML_EXPERT_CACHE

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);

    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

#ifdef GGML_EXPERT_CACHE
                    uint64_t n_tokens = ids_tensor->ne[1];

                    static int last_layer = -1;
                    int layer_id = -1;
                    // NOTE: this will probably cache non-expert tensors aswell if they're not all offloaded to the GPU. fix later.
                    if (strncmp(input->name, "blk.", 4) == 0) {
                        char* end;
                        layer_id = (int)strtol(input->name + 4, &end, 10);
                    }

                    if ((size_t)layer_id >= g_expert_caches.size()) {
                        g_expert_caches.resize(layer_id + 1);
                    }

                    if (layer_id < g_last_layer_id) {
                        for (auto& layer_map : g_expert_caches) {
                            for (auto& [name, c] : layer_map) {
#ifdef GGML_EXPERT_RAM_CACHE
                                c.disk_weights_copied = false;
                                c.has_n1_prefetch = false;
                                for (auto *dr : c.disk_ready_experts) delete dr;
                                c.disk_ready_experts.clear();
                                for (auto *dr : c.disk_inflight_experts) dr->cache = nullptr;
                                c.disk_inflight_experts.clear();
#endif
                            }
                        }
                    }
                    g_last_layer_id = layer_id;

                    auto& layer_cache_map = g_expert_caches[layer_id];
                    ggml_expert_tensor_cache* cache = nullptr;

                    auto it = layer_cache_map.find(input->name);
                    if (it == layer_cache_map.end()) {
                        ggml_expert_tensor_cache new_cache = {};
                        layer_cache_map[input->name] = new_cache;
                        cache = &layer_cache_map[input->name];
                    } else {
                        cache = &it->second;
                    }

                    if (!cache->initialized) {
                        ggml_expert_cache_init(*cache, layer_id, input->name,
                                n_expert, expert_size, split_backend, input);
                    }

                    if (input_cpy->buffer == cache->cache_buffer) {
                        input_cpy->data = cache->orig_input_cpy_d_buffer;
                        input_cpy->buffer = cache->orig_input_cpy_buffer;
                        const int64_t tmp_idx = node->op == GGML_OP_MUL_MAT_ID ? 2 : 1;
                        input_cpy->ne[tmp_idx] = n_expert;
                        input_cpy->nb[3] = n_expert * expert_size;

                        cache->orig_input_cpy_d_buffer = NULL;
                        cache->orig_input_cpy_buffer = NULL;

                        split->graph.uid = ggml_graph_next_uid();
                    }

                    // copy expert from cache to input_cpy
                    auto copy_cached_expert = [&](int32_t exp_id) {
                        if (n_tokens == 1) {
                            cache->usage_counts[exp_id]++;
                            if (cache->usage_counts[exp_id] > cache->max_usage_count) {
                                cache->max_usage_count = cache->usage_counts[exp_id];
                            }
                            cache->slot_access[exp_id] = g_access_counter++;
                            g_cache_hits++;
                        }

                        ggml_tensor cache_dst = {};
                        cache_dst.data = (char*)input_cpy->data + exp_id * expert_size;
                        cache_dst.buffer = input_cpy->buffer;
                        cache_dst.ne[0] = input_cpy->ne[0]; // input_cpy is initialized properly already
                        cache_dst.ne[1] = input_cpy->ne[1];
                        cache_dst.ne[2] = 1;
                        cache_dst.ne[3] = 1;
                        cache_dst.type = input_cpy->type;
                        cache_dst.nb[0] = input_cpy->nb[0];
                        cache_dst.nb[1] = input_cpy->nb[1];
                        cache_dst.nb[2] = cache->expert_size;
                        cache_dst.nb[3] = input_cpy->nb[3];

                        ggml_tensor cache_src = {};
                        cache_src.data = (char*)cache->d_cache_buffer + cache->expert_id_of_cached[exp_id] * expert_size;
                        cache_src.buffer = cache->cache_buffer;
                        cache_src.ne[0] = input_cpy->ne[0];
                        cache_src.ne[1] = input_cpy->ne[1];
                        cache_src.ne[2] = 1;
                        cache_src.ne[3] = 1;
                        cache_src.type = input_cpy->type;
                        cache_src.nb[0] = input_cpy->nb[0];
                        cache_src.nb[1] = input_cpy->nb[1];
                        cache_src.nb[2] = expert_size;
                        cache_src.nb[3] = input_cpy->nb[3];

                        ggml_backend_tensor_copy_async(split_backend, split_backend, &cache_src, &cache_dst);
                    };

                    // copy expert from input_cpy to cache
                    // TODO: make this more generic and only keep one of the 2 similiar lambdas. only need exp_id here.
                    auto copy_expert_to_cache = [&](uint32_t exp_id, uint32_t cache_slot) {
                        //fprintf(stdout, "tensor '%s' copying expert %d to cache from input_cpy\n", input->name, exp_id);
                        //fflush(stdout);
                        //cache->usage_counts[exp_id]++; // this isn't a use. this use was already tracked in copy_experts
                        ggml_tensor cache_dst = {};
                        cache_dst.data = (char*)cache->d_cache_buffer + cache_slot * expert_size;
                        cache_dst.buffer = cache->cache_buffer;
                        cache_dst.ne[0] = input_cpy->ne[0]; // input_cpy is initialized properly already
                        cache_dst.ne[1] = input_cpy->ne[1];
                        cache_dst.ne[2] = 1;
                        cache_dst.ne[3] = 1;
                        cache_dst.type = input_cpy->type;
                        cache_dst.nb[0] = input_cpy->nb[0];
                        cache_dst.nb[1] = input_cpy->nb[1];
                        cache_dst.nb[2] = cache->expert_size;
                        cache_dst.nb[3] = input_cpy->nb[3];

                        ggml_tensor cache_src = {};
                        cache_src.data = (char*)input_cpy->data + exp_id * expert_size;
                        cache_src.buffer = input_cpy->buffer;
                        cache_src.ne[0] = input_cpy->ne[0];
                        cache_src.ne[1] = input_cpy->ne[1];
                        cache_src.ne[2] = 1;
                        cache_src.ne[3] = 1;
                        cache_src.type = input_cpy->type;
                        cache_src.nb[0] = input_cpy->nb[0];
                        cache_src.nb[1] = input_cpy->nb[1];
                        cache_src.nb[2] = expert_size;
                        cache_src.nb[3] = input_cpy->nb[3];

                        ggml_backend_tensor_copy_async(split_backend, split_backend, &cache_src, &cache_dst);
                    };

#endif // GGML_EXPERT_CACHE

#ifdef GGML_EXPERT_CACHE
#ifdef GGML_EXPERT_RAM_CACHE
                    auto io_uring_copy_experts = [&](int32_t *experts, int nof_experts) {
                        if (n_tokens == 1) {
                            // update expert usage counts
                            for (int i = 0; i < nof_experts; i++) {
                                int exp_id = experts[i];

                                cache->usage_counts[exp_id]++;
                                if (cache->usage_counts[exp_id] > cache->max_usage_count) {
                                    cache->max_usage_count = cache->usage_counts[exp_id];
                                }
                                cache->slot_access[exp_id] = g_access_counter++;
                                g_cache_misses++;
                            }
                        }

                        std::vector<uint32_t> experts_vec;
                        std::vector<uint32_t> ram_experts_vec;

                        for (int i = 0; i < nof_experts; i++) {
                            int exp_id = experts[i];

                            int slot = cache->staging_slot_of_expert_id[exp_id];
                            size_t dst_offset = exp_id * expert_size;
                            size_t copy_size = expert_size;

                            if (slot != -1) {
                                if (n_tokens == 1) g_ram_hits++;
                                ram_experts_vec.push_back(exp_id);
                            } else {
                                if (n_tokens == 1) g_disk_misses++;
                                experts_vec.push_back(exp_id);
                            }
                        }

                        if (experts_vec.size() > 0) {
                            copy_expert_from_disk(split_backend, input_cpy, cache, &experts_vec, false);
                        }

                        for (uint32_t exp_id : ram_experts_vec) {
                            int slot = cache->staging_slot_of_expert_id[exp_id];
                            void *src = (char*)cache->pinned_staging + slot * expert_size;
                            size_t dst_offset = exp_id * expert_size;
                            ggml_backend_tensor_set_async(split_backend, input_cpy,
                                    src, dst_offset, expert_size);
                        }
                    };
#else
                    auto io_uring_copy_experts = [&](int32_t *experts, int nof_experts) {
                        if (n_tokens == 1) {
                            for (int i = 0; i < nof_experts; i++) {
                                int exp_id = experts[i];
                                cache->usage_counts[exp_id]++;
                                if (cache->usage_counts[exp_id] > cache->max_usage_count) {
                                    cache->max_usage_count = cache->usage_counts[exp_id];
                                }
                                cache->slot_access[exp_id] = g_access_counter++;
                                g_cache_misses++;
                            }
                        }

                        const size_t padding = std::min<size_t>(cache->expert_size, 512);
                        for (int i = 0; i < nof_experts; i++) {
                            int32_t exp_id = experts[i];
                            size_t expert_offset = exp_id * cache->expert_size;
                            size_t copy_size = cache->expert_size;
                            if (exp_id < (int32_t)cache->n_experts - 1) {
                                copy_size += padding;
                            }
                            ggml_backend_tensor_set_async(split_backend,
                                input_cpy,
                                (const uint8_t *)input->data + expert_offset, expert_offset,
                                copy_size);
                        }
                    };
#endif
#endif

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
#ifdef GGML_EXPERT_CACHE
                        if (n_tokens == 1) {
                            // update usage counts
                            for (int exp_id = first_id; exp_id <= last_id; exp_id++) {
                                cache->usage_counts[exp_id]++;
                                if (cache->usage_counts[exp_id] > cache->max_usage_count) {
                                    cache->max_usage_count = cache->usage_counts[exp_id];
                                }
                                cache->slot_access[exp_id] = g_access_counter++;
                                g_cache_misses++;
                            }
                        }

                        const size_t padding = std::min<size_t>(expert_size, 512);
#ifdef GGML_EXPERT_RAM_CACHE
                        std::vector<uint32_t> experts;
                        for (int exp_id = first_id; exp_id <= last_id; exp_id++) {
                            int slot = cache->staging_slot_of_expert_id[exp_id];
                            size_t dst_offset = exp_id * expert_size;
                            size_t copy_size = expert_size;

                            if (slot != -1) {
                                if (n_tokens == 1) g_ram_hits++;
                                void *src = (char*)cache->pinned_staging + slot * expert_size;
                                ggml_backend_tensor_set_async(split_backend, input_cpy,
                                    src, dst_offset, copy_size);
                            } else {
                                if (n_tokens == 1) g_disk_misses++;
                                if (exp_id < n_expert - 1) {
                                    copy_size += padding;
                                }
                                experts.push_back(exp_id);
                                //copy_expert_from_disk(split_backend, input_cpy, cache, exp_id);
                                // not in RAM cache, copy from mmap area
                                //ggml_backend_tensor_set_async(split_backend, input_cpy,
                                //    (const uint8_t *)input->data + dst_offset, dst_offset,
                                //    copy_size);
                            }
                        }
                        if (experts.size() > 0) {
                            copy_expert_from_disk(split_backend, input_cpy, cache, &experts);
                        }
#else
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            expert_size_copy + padding_end);
#endif // GGML_EXPERT_RAM_CACHE
#else
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                        // this is necessary for MMQ in the CUDA backend
                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            expert_size_copy + padding_end);
#endif
                    };

#ifdef GGML_EXPERT_CACHE

                    int32_t used_cached_experts_buf[512];
                    int32_t used_non_cached_experts_buf[512];
                    int n_used_cached = 0;
                    int n_used_non_cached = 0;

                    for (int i = 0; i < n_expert; i++) {
                        if (ggml_bitset_get(used_ids.data(), i)) {
                            if (cache->expert_id_of_cached[i] == -1) {
                                used_non_cached_experts_buf[n_used_non_cached++] = i;
                            } else {
                                used_cached_experts_buf[n_used_cached++] = i;
                            }
                        }
                    }

                    // this is supposed to be a list of the DISK resident experts
                    std::vector<uint32_t> non_cached_experts_vec;
                    for (int i = 0; i < n_used_non_cached; i++) {
#ifdef GGML_EXPERT_RAM_CACHE
                        if (cache->staging_slot_of_expert_id[used_non_cached_experts_buf[i]] == -1) {
                            non_cached_experts_vec.push_back(used_non_cached_experts_buf[i]);
                        }
#else
                        non_cached_experts_vec.push_back(used_non_cached_experts_buf[i]);
#endif
                    }


                    int32_t * used_cached_experts = used_cached_experts_buf;
                    int32_t * used_non_cached_experts = used_non_cached_experts_buf;

                    if (n_used_non_cached > 0) {

                        ggml_backend_synchronize(split_backend);
                        io_uring_copy_experts(used_non_cached_experts_buf, n_used_non_cached);

                        // overlap H->D copies of cached experts with disk reads still in flight
                        for (int ci = 0; ci < n_used_cached; ci++) {
                            int cached_exp = used_cached_experts[ci];
                            copy_cached_expert(cached_exp);
                        }

                        last_layer = -1;
                    } else {
                        cache->orig_input_cpy_d_buffer = input_cpy->data;
                        cache->orig_input_cpy_buffer = input_cpy->buffer;
                        input_cpy->data = cache->d_cache_buffer;
                        input_cpy->buffer = cache->cache_buffer;
                        const int64_t tmp_idx = node->op == GGML_OP_MUL_MAT_ID ? 2 : 1;
                        input_cpy->ne[tmp_idx] = cache->max_cached;
                        input_cpy->nb[3] = cache->max_cached * expert_size;
                        split->graph.uid = ggml_graph_next_uid();

                        if (last_layer != layer_id) {
                            // rewrite used_ids tensor: map original expert IDs to cached indices
                            for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                                for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                    int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                    int32_t orig_id = id;
                                    int32_t cached_id = cache->expert_id_of_cached[orig_id];
                                    GGML_ASSERT(cached_id != -1);
                                    if (n_tokens == 1) {
                                        cache->usage_counts[orig_id]++;
                                        if (cache->usage_counts[orig_id] > cache->max_usage_count) {
                                            cache->max_usage_count = cache->usage_counts[orig_id];
                                        }
                                        cache->slot_access[orig_id] = g_access_counter++;
                                        g_cache_hits++;
                                    }
                                    ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)] = cached_id;
                                }
                            }
                            //fprintf(stdout, "tensor '%s' new cached mapping: (%d)\n", cache->tensor_name.c_str(), last_layer);
                            ggml_backend_tensor_set_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                            ggml_backend_synchronize(ids_backend);
                        } else {
                            //fprintf(stdout, "tensor '%s' using same mapping as last layer (%d)\n\n", cache->tensor_name.c_str(), last_layer);
                            // still need to update the usage counts for these. bad things happen if we dont
                            // maybe move this after the else so both paths do it at the same place. less error prone.
                            for (int ci = 0; ci < n_used_cached; ci++) {
                                int used_exp = used_cached_experts[ci];
                                if (n_tokens == 1) {
                                    cache->usage_counts[used_exp]++;
                                    if (cache->usage_counts[used_exp] > cache->max_usage_count) {
                                        cache->max_usage_count = cache->usage_counts[used_exp];
                                    }
                                    cache->slot_access[used_exp] = g_access_counter++;
                                    g_cache_hits++;
                                }
                            }
                        }
                        last_layer = layer_id;
                    }

#else // not compiled with expert cache

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
#endif // NO GGML_EXPERT_CACHE

#ifdef GGML_EXPERT_CACHE
                    // now perform eviction and insertion.
                    if (n_tokens == 1) {
#if (GGML_EXPERT_CACHE_STRATEGY == GGML_EXPERT_CACHE_STRATEGY_LFU)
                        // manual LFU implementation
                        auto find_eviction_and_cache = [&](int expid) -> int {
                            if (cache->n_cached < cache->max_cached) {
                                // if number of entries in the cache is less than max slots of cache, return cache slot of the next free slot.
                                //fprintf(stdout, "using first empty slot %d\n", cache->n_cached);
                                return cache->n_cached;
                            }

                            uint64_t lowest_count = -1; // highest possible U64 value. all should be lower than this.
                            uint32_t lowest_count_slot = -1;

                            // now go through all cache entries and check their scores
                            // using max_cached or n_cached should be the same thing here because of the conditional above.
                            // should only reach here when n_cached = max_cached
                            for (uint32_t cache_idx = 0; cache_idx < cache->max_cached; cache_idx++) {
                                // get the expert id of this cache slot
                                int32_t remapped_expert_id = cache->cached_expert_ids[cache_idx];

                                // get the score for this cache index
                                uint64_t cached_score = cache->usage_counts[remapped_expert_id];

                                if (cached_score < lowest_count) {
                                    lowest_count = cached_score;
                                    lowest_count_slot = cache_idx;
                                }
                            }

                            // get the score of the expert we're comparing against
                            uint64_t exp_score = cache->usage_counts[expid];

                            // now compare the lowest score with the new experts score two
                            if (exp_score > lowest_count) {
                                // the score was greater than this entry's score.
                                // return the index of the cache slot `expid` should replace
                                //fprintf(stdout, "returning cache idx %d\n", lowest_count_slot);
                                return lowest_count_slot;
                            }
                            return -1;
                        };
#elif (GGML_EXPERT_CACHE_STRATEGY == GGML_EXPERT_CACHE_STRATEGY_LRU)
                        // doesn't need expid since this use will always be more recent than the one in the cache.
                        // just finds the lowest slot_access value of all the cached entries
                        auto find_eviction_and_cache = [&](int /* expid */) -> int {
                            uint64_t min_slot_access = UINT64_MAX;
                            int evict_idx = -1;
                            for (int32_t cache_idx = 0; cache_idx < cache->max_cached; cache_idx++) {
                                if (cache_idx >= cache->n_cached) {
                                    // if there's an available slot just take it.
                                    return cache_idx;
                                }
                                int remapped_expert_id = cache->cached_expert_ids[cache_idx];
                                if (remapped_expert_id < 0) {
                                    continue;
                                }
                                if (cache->slot_access[remapped_expert_id] < min_slot_access) {
                                    min_slot_access = cache->slot_access[remapped_expert_id];
                                    evict_idx = cache_idx;
                                }
                            }
                            return evict_idx;
                        };
#elif (GGML_EXPERT_CACHE_STRATEGY == GGML_EXPERT_CACHE_STRATEGY_LFU_AGING)
                        auto find_eviction_and_cache = [&](int exp_id) -> int {
                            if (cache->n_cached < cache->max_cached) {
                                return (int)cache->n_cached;
                            }
                            bool need_aging = cache->max_usage_count > GGML_EXPERT_AGING_THRESHOLD;
                            if (need_aging) {
                                for (int32_t i = 0; i < (int32_t)cache->n_experts; i++) {
                                    cache->usage_counts[i] >>= 1;
                                }
                                cache->max_usage_count >>= 1;
                            }
                            uint64_t lowest_count = UINT64_MAX;
                            uint32_t lowest_count_slot = 0;
                            for (uint32_t cache_idx = 0; cache_idx < cache->max_cached; cache_idx++) {
                                int32_t remapped_expert_id = cache->cached_expert_ids[cache_idx];
                                if (remapped_expert_id < 0) continue;
                                if (cache->usage_counts[remapped_expert_id] < lowest_count) {
                                    lowest_count = cache->usage_counts[remapped_expert_id];
                                    lowest_count_slot = cache_idx;
                                }
                            }
                            uint64_t exp_score = cache->usage_counts[exp_id];
                            if (exp_score > lowest_count) {
                                return lowest_count_slot;
                            }
                            return -1;
                        };
#elif (GGML_EXPERT_CACHE_STRATEGY == GGML_EXPERT_CACHE_STRATEGY_LFU_AGING_GUARDED)
                        auto find_eviction_and_cache = [&](int expid) -> int {
                            if (cache->n_cached < cache->max_cached) {
                                return (int)cache->n_cached;
                            }
                            bool need_aging = cache->max_usage_count > GGML_EXPERT_AGING_THRESHOLD;
                            if (need_aging) {
                                for (int32_t i = 0; i < (int32_t)cache->n_experts; i++) {
                                    cache->usage_counts[i] >>= 1;
                                }
                                cache->max_usage_count >>= 1;
                            }
                            uint64_t exp_usage = cache->usage_counts[expid];
                            if (exp_usage == 0) return -1;

                            uint64_t lowest_count = UINT64_MAX;
                            uint32_t lowest_count_slot = 0;
                            for (uint32_t cache_idx = 0; cache_idx < cache->max_cached; cache_idx++) {
                                int32_t remapped_expert_id = cache->cached_expert_ids[cache_idx];
                                if (remapped_expert_id < 0) continue;
                                if (cache->usage_counts[remapped_expert_id] < lowest_count) {
                                    lowest_count = cache->usage_counts[remapped_expert_id];
                                    lowest_count_slot = cache_idx;
                                }
                            }
                            if (exp_usage > lowest_count) {
                                return lowest_count_slot;
                            }
                            return -1;
                        };
#else
                        static_assert(false, "Invalid GGML_EXPERT_CACHE_STRATEGY");
#endif
                        //fprintf(stdout, "[expert cache] doing eviction now\n");
                        // TODO: use built up stack array here instead will reduce loop count from 256 -> GGML_EXPERT_CACHE_MAX
                        if (n_used_non_cached > 0) {
#ifdef GGML_EXPERT_RAM_CACHE
                            struct eviction_record {
                                int evict_idx;
                                int expert_id;
                                int remapped_expert_id;
                            };
                            eviction_record evictions_buf[512];
                            int n_evictions = 0;

                            for (int expert_id = 0; expert_id < n_expert; expert_id++) {
                                uint32_t is_expert_used = ggml_bitset_get(used_ids.data(), expert_id);
                                if (!is_expert_used) continue;
                                if (cache->expert_id_of_cached[expert_id] != -1) continue;

                                int evict_idx = find_eviction_and_cache(expert_id);
                                if (evict_idx >= 0) {
                                    int remapped_expert_id = cache->cached_expert_ids[evict_idx];

                                    if (cache->staging_slot_of_expert_id[expert_id] != -1) {
                                        int slot = cache->staging_slot_of_expert_id[expert_id];
                                        cache->staging_slot_of_expert_id[expert_id] = -1;
                                        cache->staging_expert_of_slot[slot] = -1;
                                        cache->free_staging_slots.push_back(slot);
                                    }

                                    if (remapped_expert_id >= 0) {
                                        cache->expert_id_of_cached[remapped_expert_id] = -1;

                                        if (cache->staging_slot_of_expert_id[remapped_expert_id] == -1
                                            && cache->staging_registered) {
                                            int free_slot = -1;

                                            if (!cache->free_staging_slots.empty()) {
                                                free_slot = cache->free_staging_slots.back();
                                                cache->free_staging_slots.pop_back();
                                            } else {
                                                int worst_staged_id = -1;
                                                uint64_t worst_count = UINT64_MAX;
                                                for (int32_t si = 0; si < cache->n_staging_slots; si++) {
                                                    int32_t eid = cache->staging_expert_of_slot[si];
                                                    if (eid < 0) continue;
                                                    if (cache->usage_counts[eid] < worst_count) {
                                                        worst_count = cache->usage_counts[eid];
                                                        worst_staged_id = eid;
                                                    }
                                                }
                                                if (worst_staged_id >= 0 && worst_count < cache->usage_counts[remapped_expert_id]) {
                                                    int victim_slot = cache->staging_slot_of_expert_id[worst_staged_id];
                                                    cache->staging_slot_of_expert_id[worst_staged_id] = -1;
                                                    cache->staging_expert_of_slot[victim_slot] = -1;
                                                    free_slot = victim_slot;

                                                }
                                            }

                                            if (free_slot >= 0) {
                                                cache->staging_slot_of_expert_id[remapped_expert_id] = free_slot;
                                                cache->staging_expert_of_slot[free_slot] = remapped_expert_id;

                                                ggml_tensor gpu_src = {};
                                                gpu_src.data = (char*)cache->d_cache_buffer + evict_idx * expert_size;
                                                gpu_src.buffer = cache->cache_buffer;
                                                gpu_src.ne[0] = input_cpy->ne[0];
                                                gpu_src.ne[1] = input_cpy->ne[1];
                                                gpu_src.ne[2] = 1;
                                                gpu_src.ne[3] = 1;
                                                gpu_src.type = input_cpy->type;
                                                gpu_src.nb[0] = input_cpy->nb[0];
                                                gpu_src.nb[1] = input_cpy->nb[1];
                                                gpu_src.nb[2] = expert_size;
                                                gpu_src.nb[3] = input_cpy->nb[3];

                                                ggml_backend_tensor_get_async(split_backend,
                                                    &gpu_src,
                                                    (char*)cache->pinned_staging + free_slot * expert_size,
                                                    0, expert_size);
                                            }
                                        }
                                    }

                                    cache->cached_expert_ids[evict_idx] = expert_id;
                                    cache->expert_id_of_cached[expert_id] = evict_idx;

                                    evictions_buf[n_evictions++] = {evict_idx, expert_id, remapped_expert_id};
                                } else if (cache->staging_slot_of_expert_id[expert_id] == -1
                                           && cache->staging_registered) {
                                    int free_slot = -1;

                                    if (!cache->free_staging_slots.empty()) {
                                        free_slot = cache->free_staging_slots.back();
                                        cache->free_staging_slots.pop_back();
                                    } else {
                                        int worst_staged_id = -1;
                                        uint64_t worst_count = UINT64_MAX;
                                        for (int32_t si = 0; si < cache->n_staging_slots; si++) {
                                            int32_t eid = cache->staging_expert_of_slot[si];
                                            if (eid < 0) continue;
                                            if (cache->usage_counts[eid] < worst_count) {
                                                worst_count = cache->usage_counts[eid];
                                                worst_staged_id = eid;
                                            }
                                        }
                                        if (worst_staged_id >= 0 && worst_count < cache->usage_counts[expert_id]) {
                                            int victim_slot = cache->staging_slot_of_expert_id[worst_staged_id];
                                            cache->staging_slot_of_expert_id[worst_staged_id] = -1;
                                            cache->staging_expert_of_slot[victim_slot] = -1;
                                            free_slot = victim_slot;
                                        }
                                    }

                                    if (free_slot >= 0) {
#ifdef GGML_EXPERT_RAM_CACHE
                    // now wait for disk reads to complete and copy to GPU
                    // eviction reads from GPU mem so all experts need to be in place at that point
                    // doesn't matter if we call this too many times, it will just instantly return.
                    if (non_cached_experts_vec.size() > 0) {
                        wait_for_reads_finish(cache, split_backend, input_cpy, &non_cached_experts_vec);
                    }
#endif // GGML_EXPERT_RAM_CACHE

                                        cache->staging_slot_of_expert_id[expert_id] = free_slot;
                                        cache->staging_expert_of_slot[free_slot] = expert_id;

                                        ggml_tensor gpu_src = {};
                                        gpu_src.data = (char*)input_cpy->data + expert_id * expert_size;
                                        gpu_src.buffer = input_cpy->buffer;
                                        gpu_src.ne[0] = input_cpy->ne[0];
                                        gpu_src.ne[1] = input_cpy->ne[1];
                                        gpu_src.ne[2] = 1;
                                        gpu_src.ne[3] = 1;
                                        gpu_src.type = input_cpy->type;
                                        gpu_src.nb[0] = input_cpy->nb[0];
                                        gpu_src.nb[1] = input_cpy->nb[1];
                                        gpu_src.nb[2] = expert_size;
                                        gpu_src.nb[3] = input_cpy->nb[3];

                                        ggml_backend_tensor_get_async(split_backend,
                                            &gpu_src,
                                            (char*)cache->pinned_staging + free_slot * expert_size,
                                            0, expert_size);
                                    }
                                }
                            }

#ifdef GGML_EXPERT_RAM_CACHE
                    // now wait for disk reads to complete and copy to GPU
                    // eviction reads from GPU mem so all experts need to be in place at that point
                    // doesn't matter if we call this too many times, it will just instantly return.
                    if (non_cached_experts_vec.size() > 0) {
                        wait_for_reads_finish(cache, split_backend, input_cpy, &non_cached_experts_vec);
                    }
#endif // GGML_EXPERT_RAM_CACHE

                            for (int ei = 0; ei < n_evictions; ei++) {
                                auto & e = evictions_buf[ei];
                                copy_expert_to_cache(e.expert_id, e.evict_idx);

                                if (e.evict_idx >= cache->n_cached) {
                                    cache->n_cached++;
                                }
                            }
#else
                            for (int expert_id = 0; expert_id < n_expert; expert_id++) {
                                uint32_t is_expert_used = ggml_bitset_get(used_ids.data(), expert_id);
                                if (!is_expert_used) continue;
                                if (cache->expert_id_of_cached[expert_id] != -1) continue;

                                int evict_idx = find_eviction_and_cache(expert_id);
                                if (evict_idx >= 0) {
                                    int remapped_expert_id = cache->cached_expert_ids[evict_idx];
                                    if (remapped_expert_id >= 0) {
                                        cache->expert_id_of_cached[remapped_expert_id] = -1;
                                    }
                                    cache->cached_expert_ids[evict_idx] = expert_id;
                                    cache->expert_id_of_cached[expert_id] = evict_idx;
                                    copy_expert_to_cache(expert_id, evict_idx);

                                    if (evict_idx >= cache->n_cached) {
                                        cache->n_cached++;
                                    }
                                }
                            }
#endif // GGML_EXPERT_RAM_CACHE
                        } else {
                            //fprintf(stdout, "not doing eviction for all-cache experts\n");
                        }
                    }
#ifdef GGML_EXPERT_RAM_CACHE
                    // now wait for disk reads to complete and copy to GPU
                    // eviction reads from GPU mem so all experts need to be in place at that point
                    // doesn't matter if we call this too many times, it will just instantly return.
                    if (non_cached_experts_vec.size() > 0) {
                        wait_for_reads_finish(cache, split_backend, input_cpy, &non_cached_experts_vec);
                    }
#endif // GGML_EXPERT_RAM_CACHE
#endif // GGML_EXPERT_CACHE
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->splits);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
