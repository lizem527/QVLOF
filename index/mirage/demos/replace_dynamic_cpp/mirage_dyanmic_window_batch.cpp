//
// Created by 许晶精 on 1/23/26.
//

/**
 * Mirage benchmark:
 *  - dynamic=0: (ef,K) grid benchmark (per-run timing + recall@K)
 *  - dynamic=1: window-level benchmark:
 *      * per-query: dump top-k id list (optional)
 *      * per-window: recall + latency statistics
 *
 * CSV inputs:
 *  - base_csv, query_csv: CSV (id,v0..v127) or plain v0..v127
 *
 * Ground truth:
 *  - computed ONCE with Kmax = max(k_list, dynamic_k), reuse prefixes
 *
 * Outputs:
 *  - --csv_path: grid output (dynamic=0)
 *  - --per_window_csv_path: window output (dynamic=1)
 *  - --ids_csv_path: per-query ids (optional, both modes)
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <climits>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <time.h>
#include <inttypes.h>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexMIRAGE.h>

static inline double now_sec(){
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return (double)tv.tv_sec + (double)tv.tv_nsec * 1e-9;
}

/*****************************************************
 * Simple "--key value" argument parser
 *****************************************************/
static std::unordered_map<std::string, std::string> parse_kv_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> m;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--help" || k == "-h") {
            m["help"] = "1";
            continue;
        }
        if (k.rfind("--", 0) != 0) {
            fprintf(stderr, "Invalid argument (expected --key value): %s\n", argv[i]);
            exit(1);
        }
        std::string key = k.substr(2);
        if (i + 1 >= argc) {
            fprintf(stderr, "Missing value for argument: %s\n", argv[i]);
            exit(1);
        }
        std::string val = argv[i + 1];
        m[key] = val;
        ++i;
    }
    return m;
}

static std::string get_arg_or(const std::unordered_map<std::string, std::string>& m,
                              const std::string& key,
                              const std::string& defv) {
    auto it = m.find(key);
    return (it == m.end()) ? defv : it->second;
}

static void print_usage(const char* prog) {
    printf("Usage:\n");
    printf("  %s --base_csv <path> --query_csv <path>\n", prog);
    printf("     [--csv_skip_header 0/1] [--csv_drop_first_col 0/1]\n");
    printf("     [--metric l2|ip]\n");
    printf("     [--rounds <int>] [--k_list <csv>] [--ef_list <csv>] [--csv_path <out.csv>]\n");
    printf("\nDynamic window mode:\n");
    printf("     [--dynamic 0/1] [--window_size <int>] [--dynamic_k <int>] [--per_window_csv_path <out.csv>]\n");
    printf("\nOptional per-query ids dump:\n");
    printf("     [--ids_csv_path <out_ids.csv>] [--ids_round_mode first|all]\n");
}

/*****************************************************
 * List parsing helpers
 *****************************************************/
static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t b = 0, e = tok.size();
        while (b < e && std::isspace((unsigned char)tok[b])) b++;
        while (e > b && std::isspace((unsigned char)tok[e - 1])) e--;
        tok = tok.substr(b, e - b);
        if (!tok.empty()) out.push_back(std::stoi(tok));
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static int max_of(const std::vector<int>& v) {
    int m = 0;
    for (int x : v) m = std::max(m, x);
    return m;
}

/*****************************************************
 * CSV helpers
 *****************************************************/
static inline std::string trim_copy(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) b++;
    while (e > b && std::isspace((unsigned char)s[e - 1])) e--;
    return s.substr(b, e - b);
}

static bool token_is_number(const std::string& tok) {
    char* endp = nullptr;
    const std::string t = trim_copy(tok);
    if (t.empty()) return false;
    std::strtod(t.c_str(), &endp);
    return endp && *endp == '\0';
}

// Read CSV matrix into contiguous float32 array: [rows, cols]
static float* csv_read_f32_2d(const char* path,
                              size_t* n0, size_t* n1,
                              bool skip_header,
                              bool drop_first_col) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        fprintf(stderr, "Failed to open CSV: %s\n", path);
        exit(1);
    }

    std::string line;
    if (skip_header) {
        if (!std::getline(fin, line)) {
            fprintf(stderr, "CSV has no data lines: %s\n", path);
            exit(1);
        }
    }

    std::vector<float> data;
    size_t dim = 0;
    size_t rows = 0;

    while (std::getline(fin, line)) {
        line = trim_copy(line);
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string tok;
        size_t col = 0;
        std::vector<float> row;
        row.reserve(256);

        while (std::getline(ss, tok, ',')) {
            tok = trim_copy(tok);
            if (tok.empty()) { col++; continue; }
            if (drop_first_col && col == 0) { col++; continue; }

            if (!token_is_number(tok)) {
                fprintf(stderr,
                        "CSV parse error (non-numeric token) in %s at row=%zu col=%zu: '%s'\n",
                        path, rows, col, tok.c_str());
                exit(1);
            }
            row.push_back((float)std::strtod(tok.c_str(), nullptr));
            col++;
        }

        if (row.empty()) continue;
        if (dim == 0) dim = row.size();
        if (row.size() != dim) {
            fprintf(stderr, "CSV inconsistent dim in %s at row=%zu: expected %zu got %zu\n",
                    path, rows, dim, row.size());
            exit(1);
        }

        data.insert(data.end(), row.begin(), row.end());
        rows++;
    }

    if (rows == 0 || dim == 0) {
        fprintf(stderr, "CSV is empty or invalid: %s\n", path);
        exit(1);
    }

    float* x = new float[rows * dim];
    std::memcpy(x, data.data(), rows * dim * sizeof(float));
    *n0 = rows;
    *n1 = dim;
    return x;
}

/*****************************************************
 * Stats helpers
 *****************************************************/
static void mean_std_sample(const std::vector<double>& xs, double* mean, double* stddev) {
    if (xs.empty()) { *mean = 0.0; *stddev = 0.0; return; }
    double s = 0.0;
    for (double v : xs) s += v;
    double m = s / (double)xs.size();
    double ss = 0.0;
    for (double v : xs) {
        double d = v - m;
        ss += d * d;
    }
    double var = (xs.size() > 1) ? (ss / (double)(xs.size() - 1)) : 0.0;
    *mean = m;
    *stddev = std::sqrt(var);
}

/*****************************************************
 * Recall helpers
 *****************************************************/
static double recall_at_k_traditional_nq(
        const faiss::idx_t* I,      // [nq, I_stride]
        int I_stride,
        int k,
        const faiss::idx_t* gt,     // [nq, gt_stride]
        int gt_stride,
        int nq) {
    if (k <= 0 || nq <= 0) return 0.0;
    if (gt_stride < k || I_stride < k) return NAN;

    const int gk = std::min(k, gt_stride);
    long long hits = 0;

    std::vector<faiss::idx_t> s;
    s.reserve((size_t)gk);

    for (int i = 0; i < nq; i++) {
        const faiss::idx_t* gt_ptr = gt + (size_t)i * (size_t)gt_stride;
        s.assign(gt_ptr, gt_ptr + (size_t)gk);
        std::sort(s.begin(), s.end());

        const faiss::idx_t* pr_ptr = I + (size_t)i * (size_t)I_stride;
        for (int j = 0; j < k; j++) {
            faiss::idx_t id = pr_ptr[j];
            if (id == (faiss::idx_t)-1) continue;
            if (std::binary_search(s.begin(), s.end(), id)) hits++;
        }
    }
    return (double)hits / ((double)nq * (double)k);
}

static double recall_one_query(
        const faiss::idx_t* Ii, int k,
        const faiss::idx_t* gti, int gt_stride) {
    int gk = std::min(k, gt_stride);
    std::vector<faiss::idx_t> s(gti, gti + gk);
    std::sort(s.begin(), s.end());

    int hit = 0;
    for (int j = 0; j < k; j++) {
        faiss::idx_t id = Ii[j];
        if (id == (faiss::idx_t)-1) continue;
        if (std::binary_search(s.begin(), s.end(), id)) hit++;
    }
    return (double)hit / (double)k;
}

/*****************************************************
 * Exact GT builder (IndexFlat) — compute once at Kmax
 *****************************************************/
static faiss::idx_t* build_exact_gt(
        const float* xb, size_t nb,
        const float* xq, size_t nq,
        int d,
        int gt_k,
        const std::string& metric) {
    if (gt_k <= 0) {
        fprintf(stderr, "[FATAL] gt_k must be positive\n");
        exit(1);
    }

    faiss::MetricType mt = faiss::METRIC_L2;
    if (metric == "ip") mt = faiss::METRIC_INNER_PRODUCT;

    faiss::IndexFlat flat(d, mt);
    flat.add((faiss::idx_t)nb, xb);

    std::vector<faiss::idx_t> I((size_t)nq * (size_t)gt_k);
    std::vector<float> D((size_t)nq * (size_t)gt_k);

    flat.search((faiss::idx_t)nq, xq, gt_k, D.data(), I.data());

    faiss::idx_t* gt = new faiss::idx_t[(size_t)nq * (size_t)gt_k];
    std::memcpy(gt, I.data(), (size_t)nq * (size_t)gt_k * sizeof(faiss::idx_t));
    return gt;
}

/*****************************************************
 * ids dump helpers
 *****************************************************/
struct IdsDumpConfig {
    std::string path;
    bool enabled = false;
    bool round_all = false; // false => first round only
};

static IdsDumpConfig parse_ids_dump_config(const std::unordered_map<std::string, std::string>& kv) {
    IdsDumpConfig cfg;
    cfg.path = get_arg_or(kv, "ids_csv_path", "");
    cfg.enabled = !cfg.path.empty();

    const std::string mode = get_arg_or(kv, "ids_round_mode", "first"); // first|all
    if (mode != "first" && mode != "all") {
        fprintf(stderr, "[FATAL] --ids_round_mode must be 'first' or 'all'\n");
        exit(1);
    }
    cfg.round_all = (mode == "all");
    return cfg;
}

static FILE* open_ids_csv_if_enabled(const IdsDumpConfig& cfg) {
    if (!cfg.enabled) return nullptr;
    FILE* f = fopen(cfg.path.c_str(), "w");
    if (!f) {
        perror("fopen ids_csv_path");
        exit(1);
    }
    fprintf(f, "ef,k,round,query,id_list\n");
    fflush(f);
    return f;
}

static void dump_ids_one_round_rowwise(
        FILE* fids,
        int ef, int kk, int rr,
        const faiss::idx_t* I,
        int nq) {
    if (!fids) return;
    for (int qi = 0; qi < nq; qi++) {
        const faiss::idx_t* ip = I + (size_t)qi * (size_t)kk;
        fprintf(fids, "%d,%d,%d,%d,\"", ef, kk, rr, qi);
        for (int r = 0; r < kk; r++) {
            if (r) fputc(' ', fids);
            fprintf(fids, "%ld", (long)ip[r]);
        }
        fprintf(fids, "\"\n");
    }
    fflush(fids);
}

// window-aware: query id = qbase + i
static void dump_ids_one_round_rowwise_with_qbase(
        FILE* fids,
        int ef, int kk, int rr,
        const faiss::idx_t* I,
        int wn,
        int qbase) {
    if (!fids) return;
    for (int i = 0; i < wn; i++) {
        const faiss::idx_t* ip = I + (size_t)i * (size_t)kk;
        int qid = qbase + i;
        fprintf(fids, "%d,%d,%d,%d,\"", ef, kk, rr, qid);
        for (int r = 0; r < kk; r++) {
            if (r) fputc(' ', fids);
            fprintf(fids, "%ld", (long)ip[r]);
        }
        fprintf(fids, "\"\n");
    }
    fflush(fids);
}

int main(int argc, char** argv) {
    double t0 = now_sec();

    auto kv = parse_kv_args(argc, argv);
    if (kv.count("help")) {
        print_usage(argv[0]);
        return 0;
    }

    // Common options
    const int rounds = std::stoi(get_arg_or(kv, "rounds", "3"));
    if (rounds <= 0) {
        fprintf(stderr, "[FATAL] --rounds must be positive\n");
        return 1;
    }

    std::string metric = get_arg_or(kv, "metric", "l2");
    if (metric != "l2" && metric != "ip") {
        fprintf(stderr, "[FATAL] --metric must be 'l2' or 'ip'\n");
        return 1;
    }

    // Grid options
    const std::string csv_out = get_arg_or(kv, "csv_path", "mirage_qps_recall_grid.csv");
    std::vector<int> k_list = parse_int_list(get_arg_or(kv, "k_list", "5,10,50,100,200,500,1000"));
    std::vector<int> efs    = parse_int_list(get_arg_or(kv, "ef_list", "20,40,100,200"));
    if (k_list.empty() || efs.empty()) {
        fprintf(stderr, "[FATAL] empty k_list or ef_list\n");
        return 1;
    }

    // Dynamic window mode (HNSW-like)
    const bool do_dynamic = (std::stoi(get_arg_or(kv, "dynamic", "0")) != 0);
    const int window_size = std::stoi(get_arg_or(kv, "window_size", "200"));
    const int dynamic_k   = std::stoi(get_arg_or(kv, "dynamic_k", "100"));
    const std::string per_window_path = get_arg_or(kv, "per_window_csv_path", "");

    // Data loading (CSV only)
    const std::string base_csv  = get_arg_or(kv, "base_csv", "");
    const std::string query_csv = get_arg_or(kv, "query_csv", "");
    if (base_csv.empty() || query_csv.empty()) {
        fprintf(stderr, "[FATAL] Provide --base_csv and --query_csv\n");
        print_usage(argv[0]);
        return 1;
    }

    // CSV options
    const bool csv_skip_header = (get_arg_or(kv, "csv_skip_header", "0") == "1");
    const bool csv_drop_first_col = (get_arg_or(kv, "csv_drop_first_col", "0") == "1");

    // Optional per-query ids
    IdsDumpConfig ids_cfg = parse_ids_dump_config(kv);

    // Determine GT length: must cover max(grid K, dynamic_k)
    const int kmax_grid = max_of(k_list);
    const int gt_k_need = std::max(kmax_grid, do_dynamic ? dynamic_k : 0);

    printf("=== Config ===\n");
    printf("metric    : %s\n", metric.c_str());
    printf("rounds    : %d\n", rounds);
    printf("dynamic   : %d\n", (int)do_dynamic);
    if (do_dynamic) {
        printf("window_size         : %d\n", window_size);
        printf("dynamic_k           : %d\n", dynamic_k);
        printf("per_window_csv_path : %s\n", per_window_path.c_str());
    } else {
        printf("csv_path (grid)     : %s\n", csv_out.c_str());
    }
    printf("base_csv  : %s\n", base_csv.c_str());
    printf("query_csv : %s\n", query_csv.c_str());
    printf("csv_skip_header   : %d\n", (int)csv_skip_header);
    printf("csv_drop_first_col: %d\n", (int)csv_drop_first_col);
    if (ids_cfg.enabled) {
        printf("ids_csv_path      : %s\n", ids_cfg.path.c_str());
        printf("ids_round_mode    : %s\n", ids_cfg.round_all ? "all" : "first");
    } else {
        printf("ids_csv_path      : (disabled)\n");
    }
    printf("============\n\n");

    if (do_dynamic && per_window_path.empty()) {
        fprintf(stderr, "[FATAL] dynamic=1 requires --per_window_csv_path\n");
        return 1;
    }
    if (do_dynamic && window_size <= 0) {
        fprintf(stderr, "[FATAL] window_size must be positive\n");
        return 1;
    }
    if (do_dynamic && dynamic_k <= 0) {
        fprintf(stderr, "[FATAL] dynamic_k must be positive\n");
        return 1;
    }

    /**********************
     * Load base + query
     **********************/
    size_t nb = 0, d_base = 0;
    float* xb = nullptr;
    size_t nq_all = 0, d_query = 0;
    float* xq = nullptr;

    printf("[%.3f s] Loading base vectors (CSV)\n", now_sec() - t0);
    xb = csv_read_f32_2d(base_csv.c_str(), &nb, &d_base, csv_skip_header, csv_drop_first_col);

    printf("[%.3f s] Loading queries (CSV)\n", now_sec() - t0);
    xq = csv_read_f32_2d(query_csv.c_str(), &nq_all, &d_query, csv_skip_header, csv_drop_first_col);

    if (d_base == 0 || d_query == 0 || d_base != d_query) {
        fprintf(stderr, "[FATAL] dim mismatch: base d=%zu, query d=%zu\n", d_base, d_query);
        return 1;
    }
    const int d = (int)d_base;

    printf("[DATA] nb=%zu d=%d, nq=%zu\n", nb, d, nq_all);

    if (nq_all > (size_t)INT_MAX) {
        fprintf(stderr, "[FATAL] nq=%zu exceeds INT_MAX; split query_csv\n", nq_all);
        return 1;
    }

    const size_t nq = nq_all;
    const int nq_i = (int)nq;

    /**********************
     * Ground Truth (GT) — once at gt_k_need
     **********************/
    const int ask_gt_k = std::min<int>(gt_k_need, (int)nb);
    if (ask_gt_k <= 0) {
        fprintf(stderr, "[FATAL] ask_gt_k <= 0\n");
        return 1;
    }

    printf("[%.3f s] Building exact GT ONCE with IndexFlat (%s), K=%d\n",
           now_sec() - t0, metric.c_str(), ask_gt_k);

    faiss::idx_t* gt = build_exact_gt(xb, nb, xq, nq, d, ask_gt_k, metric);
    const int gt_stride = ask_gt_k;
    printf("[GT] built shape: (%zu, %d)\n", nq, gt_stride);

    /**********************
     * Build Mirage once
     **********************/
    printf("\n==== Build Mirage once ====\n");
    auto* mir = new faiss::IndexMirage(d);
    mir->verbose = true;

    // your mirage params (keep as-is)
    mir->mirage.S = 20;
    mir->mirage.R = 5;
    mir->mirage.iter = 10;
    mir->hierarchy.hnsw.efConstruction = 50;

    printf("[%.3f s] Adding %zu base vectors\n", now_sec() - t0, nb);
    double tb1 = now_sec();
    mir->add((faiss::idx_t)nb, xb);
    double tb2 = now_sec();
    printf("Build done in %.1fs\n", tb2 - tb1);

    // ids csv (optional)
    FILE* fids = open_ids_csv_if_enabled(ids_cfg);

    /**********************
     * Warmup
     **********************/
    {
        int warm_k = std::min<int>(10, (int)nb);
        std::vector<faiss::idx_t> Iw((size_t)nq * (size_t)warm_k);
        std::vector<float> Dw((size_t)nq * (size_t)warm_k);
        mir->hierarchy.hnsw.efSearch = efs.front();
        mir->search((faiss::idx_t)nq, xq, warm_k, Dw.data(), Iw.data());
        printf("[%.3f s] Warmup done\n", now_sec() - t0);
    }

    /**********************
     * dynamic=1: window-level benchmark
     **********************/
    if (do_dynamic) {
        printf("\n==== Dynamic window-level benchmark (Mirage) ====\n");

        const int kk = std::min<int>(dynamic_k, (int)nb);
        if (kk > gt_stride) {
            fprintf(stderr, "[FATAL] dynamic_k=%d > gt_stride=%d\n", kk, gt_stride);
            return 1;
        }

        FILE* fpw = fopen(per_window_path.c_str(), "w");
        if (!fpw) {
            perror("fopen per_window_csv_path");
            return 1;
        }
        fprintf(fpw,
                "ef,k,window_id,q_begin,q_end,count,"
                "lat_ms_mean,lat_ms_std,sec_mean,sec_std,"
                "recall_mean,recall_std,rounds\n");
        fflush(fpw);

        const int nwin = (nq_i + window_size - 1) / window_size;

        for (int ef : efs) {
            mir->hierarchy.hnsw.efSearch = ef;

            for (int w = 0; w < nwin; w++) {
                int qb = w * window_size;
                int qe = std::min(qb + window_size, nq_i);
                int wn = qe - qb;

                std::vector<faiss::idx_t> Iw((size_t)wn * (size_t)kk);
                std::vector<float>        Dw((size_t)wn * (size_t)kk);

                std::vector<double> secs_w, lats_w;
                secs_w.reserve((size_t)rounds);
                lats_w.reserve((size_t)rounds);

                double rec_mean = 0.0;
                double rec_std  = 0.0;

                for (int rr = 0; rr < rounds; rr++) {
                    double t1 = now_sec();
                    mir->search(
                            (faiss::idx_t)wn,
                            xq + (size_t)qb * (size_t)d,
                            kk,
                            Dw.data(),
                            Iw.data()
                    );
                    double t2 = now_sec();

                    double sec = t2 - t1;
                    double lat_ms = (sec > 0) ? (sec * 1000.0 / (double)wn) : 0.0;

                    // per-query ids dump
                    if (fids) {
                        bool do_dump = ids_cfg.round_all ? true : (rr == 0);
                        if (do_dump) {
                            dump_ids_one_round_rowwise_with_qbase(
                                    fids, ef, kk, rr, Iw.data(), wn, qb
                            );
                        }
                    }

                    secs_w.push_back(sec);
                    lats_w.push_back(lat_ms);

                    // recall only once per window (rr==0), like your HNSW version
                    if (rr == 0) {
                        double rec_sum = 0.0;
                        for (int i = 0; i < wn; i++) {
                            const faiss::idx_t* Ii = Iw.data() + (size_t)i * (size_t)kk;
                            const faiss::idx_t* gti = gt + (size_t)(qb + i) * (size_t)gt_stride;
                            rec_sum += recall_one_query(Ii, kk, gti, gt_stride);
                        }
                        rec_mean = rec_sum / (double)wn;
                        rec_std  = 0.0;
                    }
                }

                double sec_mean, sec_std, lat_mean, lat_std;
                mean_std_sample(secs_w, &sec_mean, &sec_std);
                mean_std_sample(lats_w, &lat_mean, &lat_std);

                fprintf(fpw,
                        "%d,%d,%d,%d,%d,%d,"
                        "%.6f,%.6f,%.6f,%.6f,"
                        "%.6f,%.6f,%d\n",
                        ef, kk,
                        w, qb, qe - 1, wn,
                        lat_mean, lat_std,
                        sec_mean, sec_std,
                        rec_mean, rec_std,
                        rounds);
                fflush(fpw);

                printf("ef=%d window=%d | R@%d=%.4f | lat=%.3f±%.3f ms/q\n",
                       ef, w, kk, rec_mean, lat_mean, lat_std);
            }
        }

        fclose(fpw);
        if (fids) fclose(fids);

        // cleanup
        delete mir;
        delete[] xb;
        delete[] xq;
        delete[] gt;

        printf("\n[%.3f s] Done. per-window CSV written to %s\n", now_sec() - t0, per_window_path.c_str());
        if (ids_cfg.enabled) {
            printf("[%.3f s] IDs CSV written to %s\n", now_sec() - t0, ids_cfg.path.c_str());
        }
        return 0;
    }

    /**********************
     * dynamic=0: grid benchmark (your original)
     **********************/
    FILE* fout = fopen(csv_out.c_str(), "w");
    if (!fout) {
        perror("fopen csv_path");
        return 1;
    }
    fprintf(fout, "ef,k,scheme,qps_mean,qps_std,lat_ms_mean,lat_ms_std,sec_mean,sec_std,recall_mean,recall_std,rounds\n");
    fflush(fout);

    for (int ef : efs) {
        mir->hierarchy.hnsw.efSearch = ef;

        for (int k : k_list) {
            const int kk = std::min<int>(k, (int)nb);
            if (kk <= 0) continue;
            if (kk > gt_stride) {
                fprintf(stderr, "[FATAL] kk=%d > gt_stride=%d\n", kk, gt_stride);
                return 1;
            }

            std::vector<double> secs, lats_ms, qpss, recs;
            secs.reserve((size_t)rounds);
            lats_ms.reserve((size_t)rounds);
            qpss.reserve((size_t)rounds);
            recs.reserve((size_t)rounds);

            std::vector<faiss::idx_t> I((size_t)nq * (size_t)kk);
            std::vector<float>        D((size_t)nq * (size_t)kk);

            for (int rr = 0; rr < rounds; rr++) {
                double t1 = now_sec();
                mir->search((faiss::idx_t)nq, xq, kk, D.data(), I.data());
                double t2 = now_sec();

                // ids dump (grid mode, query id = 0..nq-1)
                if (fids) {
                    if (ids_cfg.round_all || rr == 0) {
                        dump_ids_one_round_rowwise(fids, ef, kk, rr, I.data(), nq_i);
                    }
                }

                double sec = t2 - t1;
                double qps = (sec > 0) ? ((double)nq / sec) : 0.0;
                double lat_ms = (sec > 0) ? (sec * 1000.0 / (double)nq) : 0.0;

                double rec = recall_at_k_traditional_nq(
                        I.data(), kk,
                        kk,
                        gt, gt_stride,
                        nq_i);

                secs.push_back(sec);
                qpss.push_back(qps);
                lats_ms.push_back(lat_ms);
                recs.push_back(rec);
            }

            double sec_mean, sec_std, qps_mean, qps_std, lat_mean, lat_std, rec_mean, rec_std;
            mean_std_sample(secs, &sec_mean, &sec_std);
            mean_std_sample(qpss, &qps_mean, &qps_std);
            mean_std_sample(lats_ms, &lat_mean, &lat_std);
            mean_std_sample(recs, &rec_mean, &rec_std);

            fprintf(fout,
                    "%d,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
                    ef, kk, "A",
                    qps_mean, qps_std,
                    lat_mean, lat_std,
                    sec_mean, sec_std,
                    rec_mean, rec_std,
                    rounds);
            fflush(fout);

            printf("ef=%d k=%d | QPS=%.1f | lat=%.3f ms/q | R@%d=%.4f\n",
                   ef, kk, qps_mean, lat_mean, kk, rec_mean);
        }
    }

    fclose(fout);
    if (fids) fclose(fids);

    // cleanup
    delete mir;
    delete[] xb;
    delete[] xq;
    delete[] gt;

    printf("\n[%.3f s] Done. CSV written to %s\n", now_sec() - t0, csv_out.c_str());
    if (ids_cfg.enabled) {
        printf("[%.3f s] IDs CSV written to %s\n", now_sec() - t0, ids_cfg.path.c_str());
    }
    return 0;
}

