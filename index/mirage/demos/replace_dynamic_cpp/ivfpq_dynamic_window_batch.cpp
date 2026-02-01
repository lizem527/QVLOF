//
// Created by 许晶精 on 1/23/26.
//

// ivfpq_dynamic_window_batch.cpp
//
// Dynamic window-level benchmark for FAISS IVFPQ (with optional refine/rerank),
// aligned with hnsw_dynamic_window_batch.cpp output style.
//
// Per-query: dump top-k id list to --ids_csv_path
// Per-window: dump recall & latency to --per_window_csv_path
//
// Args (important):
//   --base_csv <path> --query_csv <path>
//   --csv_skip_header 0/1 --csv_drop_first_col 0/1
//   --metric l2|ip
//   --rounds <int>
//   --dynamic 0/1
//   --window_size <int>
//   --dynamic_k <int>
//   --nprobe_list <csv>            (kept variable name "ef" in output for compatibility)
//   --nlist <int> --m <int> --nbits <int> --train_size <int>
//   --refine_r <int>               (>=1, >1 enables exact rerank on top k*refine_r candidates)
//   --per_window_csv_path <path>   (optional)
//   --ids_csv_path <path>          (optional)
//   --ids_round_mode first|all     (default first)
//
// Output per_window CSV columns:
//   ef,k,window_id,q_begin,q_end,count,lat_ms_mean,lat_ms_std,sec_mean,sec_std,recall_mean,recall_std,rounds
//
// Output ids CSV columns:
//   ef,k,round,query,id_list
//
// Notes:
// - Here "ef" column means "nprobe" for IVFPQ, kept for compatibility with your HNSW scripts.
// - recall is computed only on rr==0 per window (same as your HNSW dynamic code): recall_std=0.

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

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/utils/distances.h>   // exact L2/IP distance

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
    printf("  %s \n", prog);
    printf("    --base_csv <path> --query_csv <path>\n");
    printf("    [--csv_skip_header 0/1] [--csv_drop_first_col 0/1]\n");
    printf("    [--metric l2|ip]\n");
    printf("    [--rounds <int>]\n");
    printf("    [--dynamic 0/1] [--window_size <int>] [--dynamic_k <int>]\n");
    printf("    [--nprobe_list <csv>]\n");
    printf("    [--nlist <int>] [--m <int>] [--nbits <int>] [--train_size <int>]\n");
    printf("    [--refine_r <int>]\n");
    printf("    [--per_window_csv_path <out.csv>]\n");
    printf("    [--ids_csv_path <out_ids.csv>] [--ids_round_mode first|all]\n");
    printf("\nNotes:\n");
    printf("- IVFPQ is trained+built once, then run dynamic windows.\n");
    printf("- For compatibility with HNSW scripts, per-window CSV uses column name 'ef' but it actually means nprobe.\n");
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
 * mean/std
 *****************************************************/
static void mean_std(const std::vector<double>& xs, double* mean, double* stddev) {
    if (xs.empty()) { *mean = 0.0; *stddev = 0.0; return; }
    double m = 0.0;
    for (double v : xs) m += v;
    m /= (double)xs.size();
    double var = 0.0;
    for (double v : xs) {
        double d = v - m;
        var += d * d;
    }
    var /= (double)xs.size();
    *mean = m;
    *stddev = std::sqrt(var);
}

/*****************************************************
 * Exact GT builder (IndexFlat)  ——compute once at Kmax
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
 * recall one query (same style as your HNSW dynamic code)
 *****************************************************/
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
 * Exact rerank (in-memory "回表") for ONE query
 *****************************************************/
static inline void refine_one_query_exact(
        const float* xb, size_t nb,
        const float* q, int d,
        const faiss::idx_t* Iref, int k_ref,
        faiss::MetricType mt,
        faiss::idx_t* Iout, float* Dout, int k) {

    std::vector<faiss::idx_t> cand;
    cand.reserve((size_t)k_ref);
    for (int i = 0; i < k_ref; ++i) {
        faiss::idx_t id = Iref[i];
        if (id < 0) continue;
        if ((size_t)id >= nb) continue;
        cand.push_back(id);
    }
    if (cand.empty()) {
        for (int i = 0; i < k; ++i) { Iout[i] = -1; Dout[i] = 0.0f; }
        return;
    }

    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());

    struct Pair { float v; faiss::idx_t id; };
    std::vector<Pair> scored;
    scored.reserve(cand.size());

    if (mt == faiss::METRIC_L2) {
        for (faiss::idx_t id : cand) {
            const float* x = xb + (size_t)id * (size_t)d;
            float dist = faiss::fvec_L2sqr(q, x, d);
            scored.push_back({dist, id});
        }
        auto cmp = [](const Pair& a, const Pair& b){ return a.v < b.v; };
        int keep = std::min<int>(k, (int)scored.size());
        if (keep <= 0) {
            for (int i = 0; i < k; ++i) { Iout[i] = -1; Dout[i] = 0.0f; }
            return;
        }
        if (keep < (int)scored.size()) {
            std::nth_element(scored.begin(), scored.begin() + keep, scored.end(), cmp);
            scored.resize(keep);
        }
        std::sort(scored.begin(), scored.end(), cmp);
        int outn = std::min<int>(keep, k);
        for (int i = 0; i < outn; ++i) { Iout[i] = scored[i].id; Dout[i] = scored[i].v; }
        for (int i = outn; i < k; ++i) { Iout[i] = -1; Dout[i] = 0.0f; }
    } else {
        for (faiss::idx_t id : cand) {
            const float* x = xb + (size_t)id * (size_t)d;
            float ip = faiss::fvec_inner_product(q, x, d);
            scored.push_back({ip, id});
        }
        auto cmp = [](const Pair& a, const Pair& b){ return a.v > b.v; };
        int keep = std::min<int>(k, (int)scored.size());
        if (keep <= 0) {
            for (int i = 0; i < k; ++i) { Iout[i] = -1; Dout[i] = 0.0f; }
            return;
        }
        if (keep < (int)scored.size()) {
            std::nth_element(scored.begin(), scored.begin() + keep, scored.end(), cmp);
            scored.resize(keep);
        }
        std::sort(scored.begin(), scored.end(), cmp);
        int outn = std::min<int>(keep, k);
        for (int i = 0; i < outn; ++i) { Iout[i] = scored[i].id; Dout[i] = scored[i].v; }
        for (int i = outn; i < k; ++i) { Iout[i] = -1; Dout[i] = 0.0f; }
    }
}

/*****************************************************
 * ids dump config (same style as HNSW dynamic code)
 *****************************************************/
struct IdsDumpConfig {
    std::string path;
    bool enabled = false;
    bool round_all = false;  // false => first round only
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
    auto args = parse_kv_args(argc, argv);
    if (args.count("help")) {
        print_usage(argv[0]);
        return 0;
    }

    // required
    const std::string base_csv  = get_arg_or(args, "base_csv", "");
    const std::string query_csv = get_arg_or(args, "query_csv", "");
    if (base_csv.empty() || query_csv.empty()) {
        fprintf(stderr, "[FATAL] --base_csv and --query_csv are required.\n");
        print_usage(argv[0]);
        return 1;
    }

    // common
    const std::string metric = get_arg_or(args, "metric", "l2");
    if (metric != "l2" && metric != "ip") {
        fprintf(stderr, "[FATAL] --metric must be 'l2' or 'ip'\n");
        return 1;
    }
    faiss::MetricType mt = (metric == "ip") ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

    const bool csv_skip_header    = (std::stoi(get_arg_or(args, "csv_skip_header", "1")) != 0);
    const bool csv_drop_first_col = (std::stoi(get_arg_or(args, "csv_drop_first_col", "1")) != 0);

    const int rounds = std::stoi(get_arg_or(args, "rounds", "3"));
    if (rounds <= 0) {
        fprintf(stderr, "[FATAL] --rounds must be positive\n");
        return 1;
    }

    // dynamic window opts (same names as HNSW dynamic code)
    const bool do_dynamic = (std::stoi(get_arg_or(args, "dynamic", "0")) != 0);
    const int window_size = std::stoi(get_arg_or(args, "window_size", "200"));
    const int dynamic_k   = std::stoi(get_arg_or(args, "dynamic_k", "20"));
    const std::string per_window_path = get_arg_or(args, "per_window_csv_path", "");

    // nprobe list (use ef variable name for compatibility)
    std::vector<int> efs = parse_int_list(get_arg_or(args, "nprobe_list", "2,4,8,16,32"));
    if (efs.empty()) {
        fprintf(stderr, "[FATAL] empty --nprobe_list\n");
        return 1;
    }

    // IVFPQ params
    int nlist     = std::stoi(get_arg_or(args, "nlist", "512"));
    int pq_m      = std::stoi(get_arg_or(args, "m", "16"));
    int pq_nbits  = std::stoi(get_arg_or(args, "nbits", "8"));
    size_t train_size = (size_t)std::stoll(get_arg_or(args, "train_size", "1000000000"));
    if (nlist <= 0 || pq_m <= 0 || pq_nbits <= 0) {
        fprintf(stderr, "[FATAL] --nlist/--m/--nbits must be positive\n");
        return 1;
    }

    // refine
    const int refine_r = std::stoi(get_arg_or(args, "refine_r", "1"));
    if (refine_r <= 0) {
        fprintf(stderr, "[FATAL] --refine_r must be >= 1\n");
        return 1;
    }

    // ids dump (per-query)
    IdsDumpConfig ids_cfg = parse_ids_dump_config(args);
    FILE* fids = open_ids_csv_if_enabled(ids_cfg);

    printf("=== Config ===\n");
    printf("mode      : CSV\n");
    printf("metric    : %s\n", metric.c_str());
    printf("rounds    : %d\n", rounds);
    printf("dynamic   : %d\n", (int)do_dynamic);
    printf("window_size : %d\n", window_size);
    printf("dynamic_k   : %d\n", dynamic_k);
    printf("refine_r    : %d\n", refine_r);
    printf("nprobe_list : ");
    for (size_t i = 0; i < efs.size(); i++) printf("%s%d", (i ? "," : ""), efs[i]);
    printf("\n");
    printf("IVFPQ: nlist=%d m=%d nbits=%d train_size=%zu\n", nlist, pq_m, pq_nbits, train_size);
    printf("base_csv  : %s\n", base_csv.c_str());
    printf("query_csv : %s\n", query_csv.c_str());
    printf("csv_skip_header   : %d\n", (int)csv_skip_header);
    printf("csv_drop_first_col: %d\n", (int)csv_drop_first_col);
    if (!per_window_path.empty()) printf("per_window_csv_path: %s\n", per_window_path.c_str());
    else printf("per_window_csv_path: (disabled)\n");
    if (ids_cfg.enabled) {
        printf("ids_csv_path      : %s\n", ids_cfg.path.c_str());
        printf("ids_round_mode    : %s\n", ids_cfg.round_all ? "all" : "first");
    } else {
        printf("ids_csv_path      : (disabled)\n");
    }
    printf("============\n\n");

    /**********************
     * Load base + query
     **********************/
    size_t nb = 0, d_base = 0;
    float* xb = nullptr;

    size_t nq_all = 0, d_query = 0;
    float* xq_all = nullptr;

    printf("[%.3f s] Loading base vectors (CSV)\n", now_sec() - t0);
    xb = csv_read_f32_2d(base_csv.c_str(), &nb, &d_base, csv_skip_header, csv_drop_first_col);

    printf("[%.3f s] Loading queries (CSV)\n", now_sec() - t0);
    xq_all = csv_read_f32_2d(query_csv.c_str(), &nq_all, &d_query, csv_skip_header, csv_drop_first_col);

    if (!xb || nb == 0 || d_base == 0) { fprintf(stderr,"[FATAL] failed to load base_csv\n"); return 1; }
    if (!xq_all || nq_all == 0 || d_query == 0) { fprintf(stderr,"[FATAL] failed to load query_csv\n"); return 1; }

    if (d_base != d_query) {
        fprintf(stderr, "[FATAL] dim mismatch: base d=%zu, query d=%zu\n", d_base, d_query);
        return 1;
    }
    const int d = (int)d_base;

    if (nq_all > (size_t)INT_MAX) {
        fprintf(stderr, "[FATAL] nq=%zu exceeds INT_MAX; split query_csv.\n", nq_all);
        return 1;
    }

    const size_t nq = nq_all;
    const int nq_i = (int)nq;
    float* xq = xq_all;
    xq_all = nullptr;

    printf("[DATA] nb=%zu d=%d, nq=%zu\n", nb, d, nq);

    /**********************
     * Ground Truth once at Kmax (here use dynamic_k)
     **********************/
    const int ask_kmax = std::min<int>(dynamic_k, (int)nb);
    if (ask_kmax <= 0) {
        fprintf(stderr, "[FATAL] dynamic_k invalid\n");
        return 1;
    }

    printf("[%.3f s] Building exact ground truth ONCE with IndexFlat (%s), K=%d\n",
           now_sec() - t0, metric.c_str(), ask_kmax);

    faiss::idx_t* gt = build_exact_gt(xb, nb, xq, nq, d, ask_kmax, metric);
    const int gt_stride = ask_kmax;
    printf("[GT] built shape: (%zu, %d)\n", nq, gt_stride);

    /**********************
     * Build IVFPQ once
     **********************/
    printf("\n==== Build IVFPQ once, then run dynamic windows ====\n");

    auto* quantizer = new faiss::IndexFlat(d, mt);
    auto* ivfpq = new faiss::IndexIVFPQ(quantizer, d, nlist, pq_m, pq_nbits, mt);
    ivfpq->verbose = true;
    ivfpq->own_fields = true;

    size_t train_n = std::min(train_size, nb);
    if (train_n < (size_t)nlist) {
        fprintf(stderr, "[FATAL] train_n=%zu < nlist=%d; refuse to train.\n", train_n, nlist);
        return 1;
    }

    printf("[%.3f s] Training IVFPQ: train_n=%zu, nlist=%d, m=%d, nbits=%d\n",
           now_sec() - t0, train_n, nlist, pq_m, pq_nbits);
    double tt0 = now_sec();
    ivfpq->train((faiss::idx_t)train_n, xb);
    printf("[%.3f s] Training done in %.1fs\n", now_sec() - t0, now_sec() - tt0);

    printf("[%.3f s] Adding %zu base vectors\n", now_sec() - t0, nb);
    double tb1 = now_sec();
    ivfpq->add((faiss::idx_t)nb, xb);
    double tb2 = now_sec();
    printf("Build done in %.1fs\n", tb2 - tb1);

    if (!do_dynamic) {
        fprintf(stderr, "[FATAL] This file is for dynamic window benchmark. Please pass --dynamic 1.\n");
        return 1;
    }

    /**********************
     * per-window CSV
     **********************/
    FILE* fpw = nullptr;
    if (!per_window_path.empty()) {
        fpw = fopen(per_window_path.c_str(), "w");
        if (!fpw) {
            perror("fopen per_window_csv_path");
            exit(1);
        }
        fprintf(fpw,
                "ef,k,window_id,q_begin,q_end,count,"
                "lat_ms_mean,lat_ms_std,sec_mean,sec_std,"
                "recall_mean,recall_std,rounds\n");
        fflush(fpw);
    }

    /**********************
     * Warmup
     **********************/
    {
        int warm_k = std::min<int>(10, ask_kmax);
        int wn = std::min(window_size, nq_i);
        std::vector<faiss::idx_t> Iw((size_t)wn * (size_t)warm_k);
        std::vector<float>        Dw((size_t)wn * (size_t)warm_k);
        ivfpq->nprobe = efs.front();
        ivfpq->search((faiss::idx_t)wn, xq, warm_k, Dw.data(), Iw.data());
        printf("[%.3f s] Warmup done\n", now_sec() - t0);
    }

    /**********************
     * Dynamic window benchmark
     **********************/
    printf("\n==== Dynamic window-level benchmark ====\n");

    const int kk = ask_kmax; // dynamic_k (clipped by nb)
    const int nwin = (nq_i + window_size - 1) / window_size;

    for (int ef : efs) {
        ivfpq->nprobe = ef;

        for (int w = 0; w < nwin; w++) {
            int qb = w * window_size;
            int qe = std::min(qb + window_size, nq_i);
            int wn = qe - qb;

            // final outputs (after optional refine)
            std::vector<faiss::idx_t> Iw((size_t)wn * (size_t)kk);
            std::vector<float>        Dw((size_t)wn * (size_t)kk);

            // coarse outputs (k_ref candidates)
            const int k_ref = std::min<int>((int)nb, kk * refine_r);
            std::vector<faiss::idx_t> Iref((size_t)wn * (size_t)k_ref);
            std::vector<float>        Dref((size_t)wn * (size_t)k_ref);

            std::vector<double> secs_w, lats_w;
            secs_w.reserve((size_t)rounds);
            lats_w.reserve((size_t)rounds);

            double rec_mean = 0.0;
            double rec_std  = 0.0;

            for (int rr = 0; rr < rounds; rr++) {
                double t1 = now_sec();

                // 1) coarse search (PQ distance)
                ivfpq->search(
                        (faiss::idx_t)wn,
                        xq + (size_t)qb * (size_t)d,
                        k_ref,
                        Dref.data(),
                        Iref.data());

                // 2) exact refine (回表) -> final top kk
                if (refine_r > 1) {
                    for (int i = 0; i < wn; ++i) {
                        const float* qvec = (xq + (size_t)(qb + i) * (size_t)d);
                        const faiss::idx_t* Ii_ref = Iref.data() + (size_t)i * (size_t)k_ref;
                        faiss::idx_t* Ii_out = Iw.data() + (size_t)i * (size_t)kk;
                        float* Di_out = Dw.data() + (size_t)i * (size_t)kk;
                        refine_one_query_exact(xb, nb, qvec, d, Ii_ref, k_ref, mt, Ii_out, Di_out, kk);
                    }
                } else {
                    // no refine: take first kk
                    for (int i = 0; i < wn; ++i) {
                        const faiss::idx_t* srcI = Iref.data() + (size_t)i * (size_t)k_ref;
                        const float* srcD = Dref.data() + (size_t)i * (size_t)k_ref;
                        faiss::idx_t* dstI = Iw.data() + (size_t)i * (size_t)kk;
                        float* dstD = Dw.data() + (size_t)i * (size_t)kk;
                        std::memcpy(dstI, srcI, (size_t)kk * sizeof(faiss::idx_t));
                        std::memcpy(dstD, srcD, (size_t)kk * sizeof(float));
                    }
                }

                double t2 = now_sec();

                double sec = t2 - t1;
                double lat_ms = (sec > 0) ? (sec * 1000.0 / (double)wn) : 0.0;

                // dump per-query ids (OPTIONAL)
                if (fids) {
                    bool do_dump = ids_cfg.round_all ? true : (rr == 0);
                    if (do_dump) {
                        dump_ids_one_round_rowwise_with_qbase(fids, ef, kk, rr, Iw.data(), wn, qb);
                    }
                }

                secs_w.push_back(sec);
                lats_w.push_back(lat_ms);

                // recall only computed on rr==0 (same as your HNSW dynamic code)
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
            mean_std(secs_w, &sec_mean, &sec_std);
            mean_std(lats_w, &lat_mean, &lat_std);

            if (fpw) {
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
            }

            printf("nprobe=%d window=%d | refine_r=%d | R@%d=%.4f±%.4f | lat=%.3f±%.3f ms/q\n",
                   ef, w, refine_r, kk, rec_mean, rec_std, lat_mean, lat_std);
        }
    }

    if (fpw) fclose(fpw);
    if (fids) fclose(fids);

    delete ivfpq;
    delete[] xb;
    delete[] xq;
    delete[] gt;

    printf("\n[%.3f s] Done.\n", now_sec() - t0);
    return 0;
}
