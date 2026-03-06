// merge_gpu.cu  (C++/NVCC-safe, stdgraph.h include YOK)

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef uint64_t block_t;

#define BLOCK_INDEX(bit_index) ((bit_index) / 64)
#define MASK(bit_index)        ((block_t)1ULL << ((bit_index) & 63))
#define TOTAL_BLOCK_NUM(n)     (BLOCK_INDEX((n) - 1) + 1)

static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] %s: %s\n", msg, cudaGetErrorString(err));
        std::abort();
    }
}

// ------------------------------------------------------------
// Kernels: merge + conflicts
// ------------------------------------------------------------
__global__ void k_merge_blocks(
    int B,
    const block_t* parent0,
    const block_t* parent1,
    block_t* child,
    block_t* pool,
    block_t* used
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    block_t merged = 0;
    if (parent0) merged |= parent0[i];
    if (parent1) merged |= parent1[i];

    block_t new_from_parents = merged & ~used[i];
    child[i] = new_from_parents | pool[i];
    used[i] |= child[i];
    pool[i] = 0;
}

__global__ void k_conflicts_per_vertex(
    int n, int B,
    const block_t* edges,     // edges[v*B + b]
    const block_t* child,     // child[b]
    int* conflict_count,      // conflict_count[v]
    unsigned long long* sum_conflicts
){
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    int vb = BLOCK_INDEX(v);
    block_t vm = MASK(v);
    if ((child[vb] & vm) == 0) {
        conflict_count[v] = 0;
        return;
    }
//_ not: popcll e bak.
    unsigned int c = 0;
    const block_t* row = edges + (size_t)v * (size_t)B;
    for (int b = 0; b < B; b++) {
        block_t inter = row[b] & child[b];
        if (inter) c += (unsigned int)__popcll((unsigned long long)inter);
    }

    conflict_count[v] = (int)c;
    if (c) atomicAdd(sum_conflicts, (unsigned long long)c);
}

// ------------------------------------------------------------
// GPU fix_conflicts (CPU algorithm mirrored)
// ------------------------------------------------------------
struct Cand {
    int conf;
    int w;
    int v;
    unsigned tie;
};

__device__ __forceinline__ unsigned tie_hash(unsigned v, unsigned iter) {
    return (v * 1664525u) ^ (iter * 1013904223u);
}

__device__ __forceinline__ Cand better(Cand a, Cand b) {
    if (b.conf > a.conf) return b;
    if (b.conf < a.conf) return a;
    if (b.w < a.w) return b;
    if (b.w > a.w) return a;
    if (b.tie < a.tie) return b;
    return a;
}

__global__ void k_find_worst_block(
    int n,
    const block_t* child,
    const int* conflict_count,
    const int* weights,
    unsigned iter,
    int* block_best_v,
    int* block_best_conf,
    int* block_best_w
){
    __shared__ Cand sh[256];

    int tid = threadIdx.x;
    int v = blockIdx.x * blockDim.x + tid;

    Cand c;
    c.conf = -1;
    c.w = 0x7fffffff;
    c.v = -1;
    c.tie = 0xffffffffu;

    if (v < n) {
        int vb = BLOCK_INDEX(v);
        block_t vm = MASK(v);
        if (child[vb] & vm) {
            c.conf = conflict_count[v];
            c.w = weights[v];
            c.v = v;
            c.tie = tie_hash((unsigned)v, iter);
        }
    }

    sh[tid] = c;
    __syncthreads();

    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (tid < offset) sh[tid] = better(sh[tid], sh[tid + offset]);
        __syncthreads();
    }

    if (tid == 0) {
        block_best_v[blockIdx.x]    = sh[0].v;
        block_best_conf[blockIdx.x] = sh[0].conf;
        block_best_w[blockIdx.x]    = sh[0].w;
    }
}

__global__ void k_find_worst_final(
    int m,
    const int* block_v,
    const int* block_conf,
    const int* block_w,
    unsigned iter,
    int* out_worst_v
){
    __shared__ Cand sh[256];
    int tid = threadIdx.x;

    Cand c;
    c.conf = -1;
    c.w = 0x7fffffff;
    c.v = -1;
    c.tie = 0xffffffffu;

    if (tid < m) {
        int v = block_v[tid];
        if (v >= 0) {
            c.conf = block_conf[tid];
            c.w    = block_w[tid];
            c.v    = v;
            c.tie  = tie_hash((unsigned)v, iter);
        }
    }

    sh[tid] = c;
    __syncthreads();

    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (tid < offset) sh[tid] = better(sh[tid], sh[tid+offset]);
        __syncthreads();
    }

    if (tid == 0) *out_worst_v = sh[0].v;
}

__global__ void k_remove_vertex_to_pool(
    int worst,
    block_t* child,
    block_t* pool,
    int* conflict_count,
    int* pool_total
){
    int wb = BLOCK_INDEX(worst);
    block_t wm = MASK(worst);

    child[wb] &= ~wm;
    pool[wb]  |=  wm;

    atomicAdd(pool_total, 1);
    conflict_count[worst] = 0;
}

__global__ void k_decrement_neighbors_from_row(
    int n, int B,
    int worst,
    const block_t* edges,
    const block_t* child,
    int* conflict_count
){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const block_t* row = edges + (size_t)worst * (size_t)B;
    block_t inter = row[b] & child[b];

    while (inter) {
        int bit = __ffsll((unsigned long long)inter) - 1;
        int u = b*64 + bit;
        if (u < n) atomicSub(&conflict_count[u], 1);
        inter &= (inter - 1);
    }
}

static void fix_conflicts_gpu_loop(
    int n, int B,
    const block_t* d_edges,
    block_t* d_child,
    block_t* d_pool,
    int* d_conf,
    int* d_total_conflicts,
    const int* d_weights,
    int* d_pool_total
){
    const int T = 256;
    int gridV = (n + T - 1) / T;

    int *d_block_v, *d_block_conf, *d_block_w, *d_worst;
    cudaCheck(cudaMalloc(&d_block_v,    gridV * sizeof(int)), "malloc d_block_v");
    cudaCheck(cudaMalloc(&d_block_conf, gridV * sizeof(int)), "malloc d_block_conf");
    cudaCheck(cudaMalloc(&d_block_w,    gridV * sizeof(int)), "malloc d_block_w");
    cudaCheck(cudaMalloc(&d_worst,      sizeof(int)), "malloc d_worst");

    unsigned iter = 0;

    while (true) {
        int total_h = 0;
        cudaCheck(cudaMemcpy(&total_h, d_total_conflicts, sizeof(int), cudaMemcpyDeviceToHost),
                  "copy total_conflicts");
        if (total_h <= 0) break;

        k_find_worst_block<<<gridV, T>>>(n, d_child, d_conf, d_weights, iter,
                                        d_block_v, d_block_conf, d_block_w);
        cudaCheck(cudaGetLastError(), "k_find_worst_block");

        if (gridV > 256) {
            std::fprintf(stderr, "[GPU] gridV=%d too large for final reduce (need multi-pass)\n", gridV);
            std::abort();
        }
        k_find_worst_final<<<1, 256>>>(gridV, d_block_v, d_block_conf, d_block_w, iter, d_worst);
        cudaCheck(cudaGetLastError(), "k_find_worst_final");

        int worst_h = -1;
        cudaCheck(cudaMemcpy(&worst_h, d_worst, sizeof(int), cudaMemcpyDeviceToHost),
                  "copy worst");
        if (worst_h < 0) break;

        int cw_h = 0;
        cudaCheck(cudaMemcpy(&cw_h, d_conf + worst_h, sizeof(int), cudaMemcpyDeviceToHost),
                  "copy conf[worst]");

        k_remove_vertex_to_pool<<<1,1>>>(worst_h, d_child, d_pool, d_conf, d_pool_total);
        cudaCheck(cudaGetLastError(), "k_remove_vertex_to_pool");

        int gridB = (B + T - 1) / T;
        k_decrement_neighbors_from_row<<<gridB, T>>>(n, B, worst_h, d_edges, d_child, d_conf);
        cudaCheck(cudaGetLastError(), "k_decrement_neighbors_from_row");

        int new_total = total_h - cw_h;
        cudaCheck(cudaMemcpy(d_total_conflicts, &new_total, sizeof(int), cudaMemcpyHostToDevice),
                  "write total_conflicts");

        iter++;
        if (iter > (unsigned)n) break;
    }

    cudaFree(d_worst);
    cudaFree(d_block_v);
    cudaFree(d_block_conf);
    cudaFree(d_block_w);
}

// ------------------------------------------------------------
// Public entry (test_merge.cu bunu çağırıyor)
// ------------------------------------------------------------
extern "C" void merge_and_fix_gpu(
    int graph_size,
    const block_t *edges_h,
    const int *weights_h,
    const block_t **parent_color_h,
    block_t *child_h,
    block_t *pool_h,
    int *pool_count_h,
    block_t *used_h,
    int *used_vertex_count_h
){
    int n = graph_size;
    int B = TOTAL_BLOCK_NUM(graph_size);

    block_t *d_edges=nullptr, *d_child=nullptr, *d_pool=nullptr, *d_used=nullptr;
    block_t *d_p0=nullptr, *d_p1=nullptr;
    int *d_conf=nullptr;
    unsigned long long *d_sum=nullptr;
    int *d_total=nullptr;
    int *d_weights=nullptr;
    int *d_pool_total=nullptr;

    cudaCheck(cudaMalloc(&d_edges, (size_t)n*(size_t)B*sizeof(block_t)), "malloc edges");
    cudaCheck(cudaMalloc(&d_child, (size_t)B*sizeof(block_t)), "malloc child");
    cudaCheck(cudaMalloc(&d_pool,  (size_t)B*sizeof(block_t)), "malloc pool");
    cudaCheck(cudaMalloc(&d_used,  (size_t)B*sizeof(block_t)), "malloc used");
    cudaCheck(cudaMalloc(&d_conf,  (size_t)n*sizeof(int)), "malloc conf");
    cudaCheck(cudaMalloc(&d_sum,   sizeof(unsigned long long)), "malloc sum");
    cudaCheck(cudaMalloc(&d_total, sizeof(int)), "malloc total");
    cudaCheck(cudaMalloc(&d_weights, (size_t)n*sizeof(int)), "malloc weights");
    cudaCheck(cudaMalloc(&d_pool_total, sizeof(int)), "malloc pool_total");

    cudaCheck(cudaMemcpy(d_edges, edges_h, (size_t)n*(size_t)B*sizeof(block_t), cudaMemcpyHostToDevice), "copy edges");
    cudaCheck(cudaMemcpy(d_pool,  pool_h,  (size_t)B*sizeof(block_t), cudaMemcpyHostToDevice), "copy pool");
    cudaCheck(cudaMemcpy(d_used,  used_h,  (size_t)B*sizeof(block_t), cudaMemcpyHostToDevice), "copy used");
    cudaCheck(cudaMemcpy(d_weights, weights_h, (size_t)n*sizeof(int), cudaMemcpyHostToDevice), "copy weights");

    if (parent_color_h && parent_color_h[0]) {
        cudaCheck(cudaMalloc(&d_p0, (size_t)B*sizeof(block_t)), "malloc p0");
        cudaCheck(cudaMemcpy(d_p0, parent_color_h[0], (size_t)B*sizeof(block_t), cudaMemcpyHostToDevice), "copy p0");
    }
    if (parent_color_h && parent_color_h[1]) {
        cudaCheck(cudaMalloc(&d_p1, (size_t)B*sizeof(block_t)), "malloc p1");
        cudaCheck(cudaMemcpy(d_p1, parent_color_h[1], (size_t)B*sizeof(block_t), cudaMemcpyHostToDevice), "copy p1");
    }

    unsigned long long zero64 = 0;
    int zero32 = 0;
    cudaCheck(cudaMemcpy(d_sum, &zero64, sizeof(zero64), cudaMemcpyHostToDevice), "zero sum");
    cudaCheck(cudaMemcpy(d_pool_total, &zero32, sizeof(zero32), cudaMemcpyHostToDevice), "zero pool_total");

    int T = 256;
    k_merge_blocks<<<(B+T-1)/T, T>>>(B, d_p0, d_p1, d_child, d_pool, d_used);
    cudaCheck(cudaGetLastError(), "merge kernel");

    k_conflicts_per_vertex<<<(n+T-1)/T, T>>>(n, B, d_edges, d_child, d_conf, d_sum);
    cudaCheck(cudaGetLastError(), "conflicts kernel");

    unsigned long long sum_conf = 0;
    cudaCheck(cudaMemcpy(&sum_conf, d_sum, sizeof(sum_conf), cudaMemcpyDeviceToHost), "copy sum_conf");
    int total_conflicts = (int)(sum_conf / 2ULL);
    cudaCheck(cudaMemcpy(d_total, &total_conflicts, sizeof(total_conflicts), cudaMemcpyHostToDevice), "set total");

    fix_conflicts_gpu_loop(n, B, d_edges, d_child, d_pool, d_conf, d_total, d_weights, d_pool_total);

    cudaCheck(cudaMemcpy(child_h, d_child, (size_t)B*sizeof(block_t), cudaMemcpyDeviceToHost), "copy child");
    cudaCheck(cudaMemcpy(pool_h,  d_pool,  (size_t)B*sizeof(block_t), cudaMemcpyDeviceToHost), "copy pool");
    cudaCheck(cudaMemcpy(used_h,  d_used,  (size_t)B*sizeof(block_t), cudaMemcpyDeviceToHost), "copy used");

    int pool_total_h = 0;
    cudaCheck(cudaMemcpy(&pool_total_h, d_pool_total, sizeof(pool_total_h), cudaMemcpyDeviceToHost), "copy pool_total");
    *pool_count_h = pool_total_h;

    // used_vertex_count_h: istersen sonra strict saydırırız
    (void)used_vertex_count_h;

    if (d_p0) cudaFree(d_p0);
    if (d_p1) cudaFree(d_p1);
    cudaFree(d_edges);
    cudaFree(d_child);
    cudaFree(d_pool);
    cudaFree(d_used);
    cudaFree(d_conf);
    cudaFree(d_sum);
    cudaFree(d_total);
    cudaFree(d_weights);
    cudaFree(d_pool_total);

    
}
extern "C" void merge_and_fix_gpu_minimal(
    int graph_size,
    const block_t *edges_h,
    const int *weights_h,
    const block_t **parent_color_h,
    block_t *child_h,
    block_t *pool_h,
    int *pool_count_h,
    block_t *used_h
){
    // test_merge eski imzayı kullanıyor -> yeni fonksiyona map ediyoruz
    int dummy_used_count = 0;
    merge_and_fix_gpu(
        graph_size,
        edges_h,
        weights_h,
        parent_color_h,
        child_h,
        pool_h,
        pool_count_h,
        used_h,
        &dummy_used_count
    );
}

