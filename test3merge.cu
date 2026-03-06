// test_merge.cu
// Hedef A doğrulama + Solution quality (fitness proxy) istatistikleri
//
// Build (WSL):
//   nvcc -O2 -std=c++17 test_merge.cu merge_gpu.cu bitea.c stdgraph.c -o test_merge
//
// Run:
//   ./test_merge

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using block_t = uint64_t;

#define BLOCK_INDEX(bit_index) ((bit_index) / 64)
#define MASK(bit_index)        ((block_t)1ULL << ((bit_index) & 63))
#define TOTAL_BLOCK_NUM(n)     (BLOCK_INDEX((n) - 1) + 1)

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

extern "C" int count_conflicts(
    int graph_size,
    const block_t *color,
    const block_t *edges,
    int *conflict_count
);

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
// Bitset helpers
// ---------------------------
static inline block_t last_block_mask(int n) {
    int r = n & 63;
    if (r == 0) return ~0ULL;
    return (1ULL << r) - 1ULL;
}

static void mask_lastblock(int n, block_t* x, int B) {
    x[B - 1] &= last_block_mask(n);
}

static int popcount_u64(uint64_t v) {
    int c = 0;
    while (v) { v &= (v - 1); c++; }
    return c;
}

static int popcount_bitset(const block_t* x, int B) {
    int s = 0;
    for (int i = 0; i < B; i++) s += popcount_u64(x[i]);
    return s;
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

static void fill_zero(block_t* x, int B) {
    std::memset(x, 0, (size_t)B * sizeof(block_t));
}

static void set_all(block_t* x, int n, int B) {
    for (int i = 0; i < B; i++) x[i] = ~0ULL;
    mask_lastblock(n, x, B);
}

static void make_random_bitset(int n, block_t* out, int B, double p) {
    fill_zero(out, B);
    for (int v = 0; v < n; v++) {
        double r = (double)rand() / (double)RAND_MAX;
        if (r < p) out[BLOCK_INDEX(v)] |= MASK(v);
    }
    mask_lastblock(n, out, B);
}

static long long weight_sum_of_set(const block_t* setbits, int n, const int* weights) {
    long long s = 0;
    for (int v = 0; v < n; v++) {
        if (setbits[BLOCK_INDEX(v)] & MASK(v)) s += (long long)weights[v];
    }
    return s;
}

// ---------------------------
// Graph generators
// edges as flat array: edges[v*B + b]
// ---------------------------
static void make_empty_graph(int n, block_t* edges, int B) {
    std::memset(edges, 0, (size_t)n * (size_t)B * sizeof(block_t));
}

static void make_complete_graph(int n, block_t* edges, int B) {
    std::memset(edges, 0, (size_t)n * (size_t)B * sizeof(block_t));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) if (j != i) {
            edges[(size_t)i * B + BLOCK_INDEX(j)] |= MASK(j);
        }
        mask_lastblock(n, &edges[(size_t)i * B], B);
    }
}

static void make_star_graph(int n, block_t* edges, int B, int center = 0) {
    std::memset(edges, 0, (size_t)n * (size_t)B * sizeof(block_t));
    for (int v = 0; v < n; v++) if (v != center) {
        edges[(size_t)center * B + BLOCK_INDEX(v)] |= MASK(v);
        edges[(size_t)v * B + BLOCK_INDEX(center)] |= MASK(center);
    }
    for (int i = 0; i < n; i++) mask_lastblock(n, &edges[(size_t)i * B], B);
}

static void make_random_graph_edges(int n, block_t* edges, int B, double p) {
    std::memset(edges, 0, (size_t)n * (size_t)B * sizeof(block_t));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double r = (double)rand() / (double)RAND_MAX;
            if (r < p) {
                edges[(size_t)i * B + BLOCK_INDEX(j)] |= MASK(j);
                edges[(size_t)j * B + BLOCK_INDEX(i)] |= MASK(i);
            }
        }
    }
    for (int i = 0; i < n; i++) mask_lastblock(n, &edges[(size_t)i * B], B);
}

// ---------------------------
// Validation
// ---------------------------
static void validate_conflict_free_or_abort(const char* tag, int n, const block_t* child, const block_t* edges) {
    std::vector<int> cc(n, 0);
    int tot = count_conflicts(n, child, edges, cc.data());
    if (tot != 0) {
        printf("\n[FAIL] %s: child_color has conflicts: total_conflicts=%d\n", tag, tot);
        for (int i = 0; i < n; i++) {
            if (cc[i] > 0 && (child[BLOCK_INDEX(i)] & MASK(i))) {
                printf("  v%d conflicts=%d\n", i, cc[i]);
            }
        }
        print_bitset_vertices("child", child, n);
        std::abort();
    }
}

// ---------------------------
// Stats accumulator (Welford)
// ---------------------------
struct RunningStats {
    long long n = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double minv = 0.0, maxv = 0.0;
    bool inited = false;

    void add(double x) {
        if (!inited) { minv = maxv = x; inited = true; }
        minv = std::min(minv, x);
        maxv = std::max(maxv, x);

        n++;
        double delta = x - mean;
        mean += delta / (double)n;
        double delta2 = x - mean;
        m2 += delta * delta2;
    }

    double var() const { return (n > 1) ? (m2 / (double)(n - 1)) : 0.0; }
    double stddev() const { return std::sqrt(var()); }
};

struct QualityStats {
    RunningStats cpu_W, gpu_W, dW, dW_pct;
    RunningStats cpu_k, gpu_k, dk;
    RunningStats cpu_unionW, gpu_unionW; // fraction will be derived
    RunningStats cpu_frac, gpu_frac;     // W / unionW

    long long worse_gpu_count = 0;
    long long better_gpu_count = 0;
    long long equal_count = 0;

    void add_case(long long Wcpu, long long Wgpu,
                  int kcpu, int kgpu,
                  long long Wunion) {
        cpu_W.add((double)Wcpu);
        gpu_W.add((double)Wgpu);
        cpu_k.add((double)kcpu);
        gpu_k.add((double)kgpu);

        double dd = (double)(Wgpu - Wcpu);
        dW.add(dd);

        double pct = 0.0;
        if (Wcpu != 0) pct = dd / (double)std::llabs(Wcpu);
        else pct = (Wgpu == 0) ? 0.0 : (dd > 0 ? 1.0 : -1.0);
        dW_pct.add(pct);

        if (dd > 0) better_gpu_count++;
        else if (dd < 0) worse_gpu_count++;
        else equal_count++;

        cpu_unionW.add((double)Wunion);
        gpu_unionW.add((double)Wunion);

        double fcpu = (Wunion != 0) ? ((double)Wcpu / (double)Wunion) : 0.0;
        double fgpu = (Wunion != 0) ? ((double)Wgpu / (double)Wunion) : 0.0;
        cpu_frac.add(fcpu);
        gpu_frac.add(fgpu);

        dk.add((double)(kgpu - kcpu));
    }

    void print_summary() const {
        printf("\n===== Solution Quality Summary (across all cases) =====\n");

        printf("Cases: %lld\n", cpu_W.n);

        printf("\nChild weight sum W = sum(weights[v] for v in child)\n");
        printf("  CPU: mean=%.2f std=%.2f min=%.0f max=%.0f\n",
               cpu_W.mean, cpu_W.stddev(), cpu_W.minv, cpu_W.maxv);
        printf("  GPU: mean=%.2f std=%.2f min=%.0f max=%.0f\n",
               gpu_W.mean, gpu_W.stddev(), gpu_W.minv, gpu_W.maxv);

        printf("\nDelta W = W_gpu - W_cpu\n");
        printf("  mean=%.2f std=%.2f min=%.2f max=%.2f\n",
               dW.mean, dW.stddev(), dW.minv, dW.maxv);
        printf("  GPU better: %lld | worse: %lld | equal: %lld\n",
               better_gpu_count, worse_gpu_count, equal_count);

        printf("\nDelta W%% = (W_gpu - W_cpu) / |W_cpu|\n");
        printf("  mean=%.4f std=%.4f min=%.4f max=%.4f\n",
               dW_pct.mean, dW_pct.stddev(), dW_pct.minv, dW_pct.maxv);

        printf("\nChild size k = |child|\n");
        printf("  CPU: mean=%.2f std=%.2f min=%.0f max=%.0f\n",
               cpu_k.mean, cpu_k.stddev(), cpu_k.minv, cpu_k.maxv);
        printf("  GPU: mean=%.2f std=%.2f min=%.0f max=%.0f\n",
               gpu_k.mean, gpu_k.stddev(), gpu_k.minv, gpu_k.maxv);

        printf("\nDelta k = |child_gpu| - |child_cpu|\n");
        printf("  mean=%.3f std=%.3f min=%.0f max=%.0f\n",
               dk.mean, dk.stddev(), dk.minv, dk.maxv);

        printf("\nNormalized quality fraction f = W_child / W_expected_union\n");
        printf("  CPU f: mean=%.4f std=%.4f min=%.4f max=%.4f\n",
               cpu_frac.mean, cpu_frac.stddev(), cpu_frac.minv, cpu_frac.maxv);
        printf("  GPU f: mean=%.4f std=%.4f min=%.4f max=%.4f\n",
               gpu_frac.mean, gpu_frac.stddev(), gpu_frac.minv, gpu_frac.maxv);

        printf("======================================================\n");
    }
};

// ---------------------------
// Case runner
// ---------------------------
struct CaseConfig {
    int n;
    std::string graph_kind; // "empty", "random", "complete", "star"
    double edge_p;          // for random only
    double bit_p_parents;   // density for parents
    double bit_p_used;      // density for used0
    double bit_p_pool;      // density for pool0
    std::string scenario;   // extra scenario name
    unsigned seed;
};

static void run_case(const CaseConfig& cfg, QualityStats& Q) {
    const int n = cfg.n;
    const int B = TOTAL_BLOCK_NUM(n);

    srand(cfg.seed);

    // Graph
    std::vector<block_t> edges((size_t)n * (size_t)B);
    if (cfg.graph_kind == "empty") make_empty_graph(n, edges.data(), B);
    else if (cfg.graph_kind == "complete") make_complete_graph(n, edges.data(), B);
    else if (cfg.graph_kind == "star") make_star_graph(n, edges.data(), B, 0);
    else make_random_graph_edges(n, edges.data(), B, cfg.edge_p);

    // Weights (harder spread)
    std::vector<int> weights(n);
    for (int i = 0; i < n; i++) {
        int r = rand() % 100;
        weights[i] = (r < 5) ? (1 + rand() % 3) : (5 + rand() % 1000);
    }

    // Bitsets
    std::vector<block_t> p0(B), p1(B), used0(B), pool0(B);
    make_random_bitset(n, p0.data(), B, cfg.bit_p_parents);
    make_random_bitset(n, p1.data(), B, cfg.bit_p_parents);
    make_random_bitset(n, used0.data(), B, cfg.bit_p_used);
    make_random_bitset(n, pool0.data(), B, cfg.bit_p_pool);

    if (cfg.scenario == "used_all") {
        set_all(used0.data(), n, B);
    } else if (cfg.scenario == "pool_all") {
        set_all(pool0.data(), n, B);
    } else if (cfg.scenario == "parents_identical") {
        std::memcpy(p1.data(), p0.data(), (size_t)B * sizeof(block_t));
    } else if (cfg.scenario == "parents_disjoint") {
        for (int i = 0; i < B; i++) p1[i] &= ~p0[i];
        mask_lastblock(n, p1.data(), B);
    } else if (cfg.scenario == "used_equals_pool") {
        std::memcpy(used0.data(), pool0.data(), (size_t)B * sizeof(block_t));
    }

    const block_t* parents[2] = { p0.data(), p1.data() };

    // expected_union = pool0 | ((p0|p1) & ~used0)
    std::vector<block_t> parents_union(B), new_from_parents(B), expected_union(B);
    bitset_or(p0.data(), p1.data(), parents_union.data(), B);
    bitset_andnot(parents_union.data(), used0.data(), new_from_parents.data(), B);
    bitset_or(pool0.data(), new_from_parents.data(), expected_union.data(), B);
    mask_lastblock(n, expected_union.data(), B);

    // used_expected = used0 | expected_union
    std::vector<block_t> used_expected(B);
    bitset_or(used0.data(), expected_union.data(), used_expected.data(), B);
    mask_lastblock(n, used_expected.data(), B);

    // CPU
    std::vector<block_t> child_cpu(B), pool_cpu(B), used_cpu(B);
    std::memcpy(pool_cpu.data(), pool0.data(), (size_t)B * sizeof(block_t));
    std::memcpy(used_cpu.data(), used0.data(), (size_t)B * sizeof(block_t));
    int pool_count_cpu = popcount_bitset(pool_cpu.data(), B);
    int used_count_cpu = popcount_bitset(used_cpu.data(), B);

    srand(12345);
    merge_and_fix(n, edges.data(), weights.data(), parents,
                  child_cpu.data(), pool_cpu.data(), &pool_count_cpu,
                  used_cpu.data(), &used_count_cpu);

    // GPU
    std::vector<block_t> child_gpu(B), pool_gpu(B), used_gpu(B);
    std::memcpy(pool_gpu.data(), pool0.data(), (size_t)B * sizeof(block_t));
    std::memcpy(used_gpu.data(), used0.data(), (size_t)B * sizeof(block_t));
    int pool_count_gpu = popcount_bitset(pool_gpu.data(), B);

    srand(12345);
    merge_and_fix_gpu_minimal(n, edges.data(), weights.data(), parents,
                              child_gpu.data(), pool_gpu.data(), &pool_count_gpu,
                              used_gpu.data());

    // Validity: conflict-free
    validate_conflict_free_or_abort("CPU", n, child_cpu.data(), edges.data());
    validate_conflict_free_or_abort("GPU", n, child_gpu.data(), edges.data());

    // Invariants:
    // 1) child & pool == 0
    std::vector<block_t> inter_cpu(B), inter_gpu(B);
    bitset_and(child_cpu.data(), pool_cpu.data(), inter_cpu.data(), B);
    bitset_and(child_gpu.data(), pool_gpu.data(), inter_gpu.data(), B);
    mask_lastblock(n, inter_cpu.data(), B);
    mask_lastblock(n, inter_gpu.data(), B);

    // 2) union == expected_union
    std::vector<block_t> union_cpu(B), union_gpu(B);
    bitset_or(child_cpu.data(), pool_cpu.data(), union_cpu.data(), B);
    bitset_or(child_gpu.data(), pool_gpu.data(), union_gpu.data(), B);
    mask_lastblock(n, union_cpu.data(), B);
    mask_lastblock(n, union_gpu.data(), B);

    // 3) used == used_expected
    mask_lastblock(n, used_cpu.data(), B);
    mask_lastblock(n, used_gpu.data(), B);

    bool cpu_disjoint_ok = bitset_is_zero(inter_cpu.data(), B);
    bool gpu_disjoint_ok = bitset_is_zero(inter_gpu.data(), B);
    bool cpu_union_ok = bitset_equal(union_cpu.data(), expected_union.data(), B);
    bool gpu_union_ok = bitset_equal(union_gpu.data(), expected_union.data(), B);
    bool cpu_used_ok  = bitset_equal(used_cpu.data(), used_expected.data(), B);
    bool gpu_used_ok  = bitset_equal(used_gpu.data(), used_expected.data(), B);

    int pool_pop_cpu = popcount_bitset(pool_cpu.data(), B);
    int pool_pop_gpu = popcount_bitset(pool_gpu.data(), B);
    bool cpu_poolcnt_ok = (pool_count_cpu == pool_pop_cpu);
    bool gpu_poolcnt_ok = (pool_count_gpu == pool_pop_gpu);

    if (!cpu_disjoint_ok || !gpu_disjoint_ok || !cpu_union_ok || !gpu_union_ok ||
        !cpu_used_ok || !gpu_used_ok || !cpu_poolcnt_ok || !gpu_poolcnt_ok) {

        printf("\n[FAIL] Invariant mismatch!\n");
        printf("  n=%d graph=%s edge_p=%.3f scenario=%s seed=%u\n",
               n, cfg.graph_kind.c_str(), cfg.edge_p, cfg.scenario.c_str(), cfg.seed);
        printf("  bit_p_parents=%.3f bit_p_used=%.3f bit_p_pool=%.3f\n",
               cfg.bit_p_parents, cfg.bit_p_used, cfg.bit_p_pool);

        print_bitset_vertices("p0", p0.data(), n);
        print_bitset_vertices("p1", p1.data(), n);
        print_bitset_vertices("used0", used0.data(), n);
        print_bitset_vertices("pool0", pool0.data(), n);

        print_bitset_vertices("expected_union", expected_union.data(), n);
        print_bitset_vertices("used_expected", used_expected.data(), n);

        print_bitset_vertices("child_cpu", child_cpu.data(), n);
        print_bitset_vertices("pool_cpu",  pool_cpu.data(),  n);
        print_bitset_vertices("used_cpu",  used_cpu.data(),  n);

        print_bitset_vertices("child_gpu", child_gpu.data(), n);
        print_bitset_vertices("pool_gpu",  pool_gpu.data(),  n);
        print_bitset_vertices("used_gpu",  used_gpu.data(),  n);

        std::abort();
    }

    // -------- Quality metrics collection --------
    long long Wcpu = weight_sum_of_set(child_cpu.data(), n, weights.data());
    long long Wgpu = weight_sum_of_set(child_gpu.data(), n, weights.data());
    int kcpu = popcount_bitset(child_cpu.data(), B);
    int kgpu = popcount_bitset(child_gpu.data(), B);
    long long Wunion = weight_sum_of_set(expected_union.data(), n, weights.data());

    Q.add_case(Wcpu, Wgpu, kcpu, kgpu, Wunion);
}

int main() {
    printf("Stress test suite for merge_and_fix (Hedef A invariants + quality stats)\n");

    const int Ns[] = { 1,2,3, 31,32,33, 63,64,65, 127,128,129, 255,256,257, 511,512, 1023,1024 };
    const std::vector<std::string> graph_kinds = {"empty", "random", "star", "complete"};
    const std::vector<std::string> scenarios = {
        "normal","used_all","pool_all","parents_identical","parents_disjoint","used_equals_pool"
    };

    struct D { double p_par; double p_used; double p_pool; };
    const D densities[] = {
        {0.05, 0.02, 0.01},
        {0.20, 0.10, 0.05},
        {0.50, 0.25, 0.10},
    };

    const double edge_ps[] = {0.0, 0.02, 0.10, 0.50, 0.90};

    unsigned base_seed = 777;
    int total = 0;
    int passed = 0;

    QualityStats Q;

    for (int ni = 0; ni < (int)(sizeof(Ns)/sizeof(Ns[0])); ni++) {
        int n = Ns[ni];

        for (const auto& gk : graph_kinds) {
            for (const auto& sc : scenarios) {
                for (const auto& d : densities) {

                    std::vector<double> eps;
                    if (gk == "random") {
                        eps.assign(edge_ps, edge_ps + (sizeof(edge_ps)/sizeof(edge_ps[0])));
                    } else {
                        eps = {0.0};
                    }

                    for (double ep : eps) {
                        if (gk == "complete" && n > 256) continue;
                        if (gk == "random" && n >= 1024 && ep >= 0.90 && d.p_par >= 0.50) continue;

                        for (int rep = 0; rep < 3; rep++) {
                            CaseConfig cfg;
                            cfg.n = n;
                            cfg.graph_kind = gk;
                            cfg.edge_p = ep;
                            cfg.bit_p_parents = d.p_par;
                            cfg.bit_p_used = d.p_used;
                            cfg.bit_p_pool = d.p_pool;
                            cfg.scenario = sc;
                            cfg.seed = base_seed + (unsigned)(total * 1315423911u) + (unsigned)rep * 2654435761u;

                            total++;
                            run_case(cfg, Q);
                            passed++;

                            if ((passed % 200) == 0) {
                                printf("  passed %d cases...\n", passed);
                                fflush(stdout);
                            }
                        }
                    }
                }
            }
        }
    }

    printf("[OK] All tests passed. total_cases=%d\n", total);
    Q.print_summary();
    return 0;
}