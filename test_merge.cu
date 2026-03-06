// test_merge.cu
// Hedef A test: CPU ve GPU birebir aynı child/pool üretmek zorunda DEĞİL.
// Ama ikisi de conflict-free olmalı ve bazı invariant'lar sağlanmalı.
//
// Build (WSL):
//   nvcc -O2 -std=c++17 test_merge.cu merge_gpu.cu bitea.c stdgraph.c -o test_merge

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cassert>

using block_t = uint64_t;

#define BLOCK_INDEX(bit_index) ((bit_index) / 64)
#define MASK(bit_index)        ((block_t)1ULL << ((bit_index) & 63))
#define TOTAL_BLOCK_NUM(n)     (BLOCK_INDEX((n) - 1) + 1)

// ---------------------------
// External functions from your CPU code
// ---------------------------

// Original CPU version (from your C file)
extern "C" void merge_and_fix(
    int graph_size,
    const block_t *edges,
    const int *weights,
    const block_t **parent_color,
    block_t *child_color,
    block_t *pool,
    int *pool_count,
    block_t *used_vertex_list,
    int *used_vertex_count
);

// From stdgraph.c
extern "C" int count_conflicts(
    int graph_size,
    const block_t *color,
    const block_t *edges,
    int *conflict_count
);

// Your GPU minimal wrapper (from your .cu where you implemented it)
extern "C" void merge_and_fix_gpu_minimal(
    int graph_size,
    const block_t *edges_h,
    const int *weights_h,
    const block_t **parent_color_h, // [2]
    block_t *child_h,
    block_t *pool_h,
    int *pool_count_h,
    block_t *used_h
);

// ---------------------------
// Helpers
// ---------------------------
static inline block_t last_block_mask(int n) {
    int r = n & 63;
    if (r == 0) return ~0ULL;
    // r is in [1..63]
    return (1ULL << r) - 1ULL;
}

static void mask_all_bitsets_lastblock(int n, block_t* x, int B) {
    x[B - 1] &= last_block_mask(n);
}

static int popcount_u64(uint64_t v) {
    // portable popcount
    int c = 0;
    while (v) { v &= (v - 1); c++; }
    return c;
}

static int popcount_bitset(const block_t* x, int B) {
    int s = 0;
    for (int i = 0; i < B; i++) s += popcount_u64(x[i]);
    return s;
}

static void make_random_bitset(int n, block_t* out, double p) {
    int B = TOTAL_BLOCK_NUM(n);
    std::memset(out, 0, B * sizeof(block_t));
    for (int v = 0; v < n; v++) {
        double r = (double)rand() / (double)RAND_MAX;
        if (r < p) {
            out[BLOCK_INDEX(v)] |= MASK(v);
        }
    }
    mask_all_bitsets_lastblock(n, out, B);
}

static void make_random_graph_edges(int n, block_t* edges_flat, double p) {
    // edges_flat layout: edges[v*B + b]
    int B = TOTAL_BLOCK_NUM(n);
    std::memset(edges_flat, 0, (size_t)n * (size_t)B * sizeof(block_t));

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double r = (double)rand() / (double)RAND_MAX;
            if (r < p) {
                // set undirected edge i<->j
                edges_flat[(size_t)i * B + BLOCK_INDEX(j)] |= MASK(j);
                edges_flat[(size_t)j * B + BLOCK_INDEX(i)] |= MASK(i);
            }
        }
    }

    // Clean last block for each row
    for (int v = 0; v < n; v++) {
        edges_flat[(size_t)v * B + (B - 1)] &= last_block_mask(n);
    }
}

static bool bitset_equal(const block_t* a, const block_t* b, int B) {
    for (int i = 0; i < B; i++) if (a[i] != b[i]) return false;
    return true;
}

static bool bitset_is_zero(const block_t* a, int B) {
    for (int i = 0; i < B; i++) if (a[i] != 0) return false;
    return true;
}

static void bitset_or(const block_t* a, const block_t* b, block_t* out, int B) {
    for (int i = 0; i < B; i++) out[i] = a[i] | b[i];
}

static void bitset_and(const block_t* a, const block_t* b, block_t* out, int B) {
    for (int i = 0; i < B; i++) out[i] = a[i] & b[i];
}

static void bitset_andnot(const block_t* a, const block_t* b, block_t* out, int B) {
    for (int i = 0; i < B; i++) out[i] = a[i] & ~b[i];
}

static void print_bitset_vertices(const char* name, const block_t* x, int n) {
    printf("%s: {", name);
    bool first = true;
    for (int v = 0; v < n; v++) {
        if (x[BLOCK_INDEX(v)] & MASK(v)) {
            if (!first) printf(", ");
            printf("%d", v);
            first = false;
        }
    }
    printf("}\n");
}

static void validate_conflict_free(const char* tag, int n, const block_t* child, const block_t* edges) {
    std::vector<int> cc(n, 0);
    int tot = count_conflicts(n, child, edges, cc.data());
    if (tot != 0) {
        printf("[FAIL] %s: child_color has conflicts: total_conflicts=%d\n", tag, tot);
        for (int i = 0; i < n; i++) {
            if (cc[i] > 0 && (child[BLOCK_INDEX(i)] & MASK(i))) {
                printf("  v%d conflicts=%d\n", i, cc[i]);
            }
        }
        std::abort();
    }
}

// ---------------------------
// One test case
// ---------------------------
static void run_one_case(int n, double edge_p, double bit_p) {
    const int B = TOTAL_BLOCK_NUM(n);

    // --- Graph + weights
    std::vector<block_t> edges((size_t)n * (size_t)B);
    make_random_graph_edges(n, edges.data(), edge_p);

    std::vector<int> weights(n);
    for (int i = 0; i < n; i++) weights[i] = 1 + (rand() % 20);

    // --- Parents, used, pool (initial)
    std::vector<block_t> p0(B), p1(B), used0(B), pool0(B);
    make_random_bitset(n, p0.data(), bit_p);
    make_random_bitset(n, p1.data(), bit_p);
    make_random_bitset(n, used0.data(), bit_p * 0.4);  // fewer used
    make_random_bitset(n, pool0.data(), bit_p * 0.2);  // fewer pool

    // enforce clean last block
    mask_all_bitsets_lastblock(n, p0.data(), B);
    mask_all_bitsets_lastblock(n, p1.data(), B);
    mask_all_bitsets_lastblock(n, used0.data(), B);
    mask_all_bitsets_lastblock(n, pool0.data(), B);

    const block_t* parents[2] = { p0.data(), p1.data() };

    // --- CPU run
    std::vector<block_t> child_cpu(B), pool_cpu(B), used_cpu(B);
    std::memcpy(pool_cpu.data(), pool0.data(), B * sizeof(block_t));
    std::memcpy(used_cpu.data(), used0.data(), B * sizeof(block_t));
    int pool_count_cpu = popcount_bitset(pool_cpu.data(), B);
    int used_count_cpu = popcount_bitset(used_cpu.data(), B);

    srand(12345);
    merge_and_fix(
        n,
        edges.data(),
        weights.data(),
        parents,
        child_cpu.data(),
        pool_cpu.data(),
        &pool_count_cpu,
        used_cpu.data(),
        &used_count_cpu
    );

    // --- GPU run
    std::vector<block_t> child_gpu(B), pool_gpu(B), used_gpu(B);
    std::memcpy(pool_gpu.data(), pool0.data(), B * sizeof(block_t));
    std::memcpy(used_gpu.data(), used0.data(), B * sizeof(block_t));
    int pool_count_gpu = popcount_bitset(pool_gpu.data(), B);

    srand(12345);
    merge_and_fix_gpu_minimal(
        n,
        edges.data(),
        weights.data(),
        parents,
        child_gpu.data(),
        pool_gpu.data(),
        &pool_count_gpu,
        used_gpu.data()
    );

    // --- Validity checks (both must be conflict-free)
    validate_conflict_free("CPU", n, child_cpu.data(), edges.data());
    validate_conflict_free("GPU", n, child_gpu.data(), edges.data());

    // ============================================================
    // Hedef A: Invariant checks 
    // ============================================================

    // expected_union = pool0 | ((p0|p1) & ~used0)
    std::vector<block_t> parents_union(B), new_from_parents(B), expected_union(B);
    bitset_or(p0.data(), p1.data(), parents_union.data(), B);
    bitset_andnot(parents_union.data(), used0.data(), new_from_parents.data(), B);
    bitset_or(pool0.data(), new_from_parents.data(), expected_union.data(), B);
    mask_all_bitsets_lastblock(n, expected_union.data(), B);

    // union_cpu/gpu = child | pool
    std::vector<block_t> union_cpu(B), union_gpu(B);
    bitset_or(child_cpu.data(), pool_cpu.data(), union_cpu.data(), B);
    bitset_or(child_gpu.data(), pool_gpu.data(), union_gpu.data(), B);
    mask_all_bitsets_lastblock(n, union_cpu.data(), B);
    mask_all_bitsets_lastblock(n, union_gpu.data(), B);

    // disjoint check: child & pool == 0
    std::vector<block_t> inter_cpu(B), inter_gpu(B);
    bitset_and(child_cpu.data(), pool_cpu.data(), inter_cpu.data(), B);
    bitset_and(child_gpu.data(), pool_gpu.data(), inter_gpu.data(), B);
    mask_all_bitsets_lastblock(n, inter_cpu.data(), B);
    mask_all_bitsets_lastblock(n, inter_gpu.data(), B);

    // used_expected = used0 | expected_union
    std::vector<block_t> used_expected(B);
    bitset_or(used0.data(), expected_union.data(), used_expected.data(), B);
    mask_all_bitsets_lastblock(n, used_expected.data(), B);

    bool cpu_union_ok = bitset_equal(union_cpu.data(), expected_union.data(), B);
    bool gpu_union_ok = bitset_equal(union_gpu.data(), expected_union.data(), B);
    bool cpu_disjoint_ok = bitset_is_zero(inter_cpu.data(), B);
    bool gpu_disjoint_ok = bitset_is_zero(inter_gpu.data(), B);
    bool cpu_used_ok = bitset_equal(used_cpu.data(), used_expected.data(), B);
    bool gpu_used_ok = bitset_equal(used_gpu.data(), used_expected.data(), B);

    int pool_pop_cpu = popcount_bitset(pool_cpu.data(), B);
    int pool_pop_gpu = popcount_bitset(pool_gpu.data(), B);
    bool cpu_poolcnt_ok = (pool_count_cpu == pool_pop_cpu);
    bool gpu_poolcnt_ok = (pool_count_gpu == pool_pop_gpu);

    if (!cpu_union_ok || !gpu_union_ok || !cpu_disjoint_ok || !gpu_disjoint_ok ||
        !cpu_used_ok || !gpu_used_ok || !cpu_poolcnt_ok || !gpu_poolcnt_ok) {

        printf("\n[FAIL] Invariant mismatch (n=%d)\n", n);
        printf(" cpu_union_ok=%d gpu_union_ok=%d\n", (int)cpu_union_ok, (int)gpu_union_ok);
        printf(" cpu_disjoint_ok=%d gpu_disjoint_ok=%d\n", (int)cpu_disjoint_ok, (int)gpu_disjoint_ok);
        printf(" cpu_used_ok=%d gpu_used_ok=%d\n", (int)cpu_used_ok, (int)gpu_used_ok);
        printf(" cpu_poolcnt_ok=%d gpu_poolcnt_ok=%d\n", (int)cpu_poolcnt_ok, (int)gpu_poolcnt_ok);

        print_bitset_vertices("expected_union", expected_union.data(), n);

        print_bitset_vertices("child_cpu", child_cpu.data(), n);
        print_bitset_vertices("pool_cpu",  pool_cpu.data(),  n);
        print_bitset_vertices("used_cpu",  used_cpu.data(),  n);

        print_bitset_vertices("child_gpu", child_gpu.data(), n);
        print_bitset_vertices("pool_gpu",  pool_gpu.data(),  n);
        print_bitset_vertices("used_gpu",  used_gpu.data(),  n);

        std::abort();
    }
}

// ---------------------------
// Main: deterministic fuzz
// ---------------------------
int main() {
    // Fixed seed for reproducibility of graph + parents
    srand(777);

    const int n = 1024;           // try also 20, 64, 128...
    const double edge_p = 0.15;  // graph density
    const double bit_p  = 0.25;  // density for parent/used/pool bitsets

    printf("Running deterministic fuzz tests...\n");

    // Run multiple cases
    for (int t = 0; t < 50; t++) {
        run_one_case(n, edge_p, bit_p);
        if ((t + 1) % 10 == 0) printf("  passed %d/50\n", t + 1);
    }

    printf("[OK] All tests passed.\n");
    return 0;
}

//Çıktı birebir aynı olmak zorunda,yanlış varsa yanlışı bulup çözmemiz gerekir.

//Hız farklarını not edelim.
// Ne kadar paraleliz şu anda bununla ilgili ufak bi rapor.
//Summarydeki farkı kontrol