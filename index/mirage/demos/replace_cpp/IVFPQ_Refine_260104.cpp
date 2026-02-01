//
// Created by 许晶精 on 12/26/25.
//
// IVFPQ query performance benchmark (Scheme A only) + optional per-query topK dump
//
// New:
//   --hits_csv <path>    (optional) dump each query's topK (returned label/id list)
//   --hits_write_round 0/1  (default 0) whether to dump for every round (1) or only rr==0 (0)
//
// Added (RERANK / REFINE):
//   --refine_r <int> (default 1)
//      - refine_r == 1: no refine (original behavior)
//      - refine_r  > 1: coarse search returns k_ref = min(nb, k * refine_r) candidates (PQ distance),
//                       then we do exact rerank by reading xb[id] (in-memory “回表”) and computing true L2/IP,
//                       output final top-k.
//
// hits_csv columns:
//   nprobe,k,round,qi,id_list
// NOTE:
//   id_list writes the returned labels (ids). With add() (no xids), ids are 0..nb-1 in add order.
//   If you later switch to add_with_ids(), this will be those provided ids (not physical layout positions).
//

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
    printf("Usage:");
    printf("  %s \n", prog);
    printf("    --base_csv <path> --query_csv <path> \n");
    printf("    [--csv_skip_header 0/1] [--csv_drop_first_col 0/1] \n");
    printf("    [--metric l2|ip] \n");
    printf("    [--rounds <int>] [--k_list <csv>] [--nprobe_list <csv>] [--csv_path <out.csv>]\n");
    printf("    [--nlist <int>] [--m <int>] [--nbits <int>]  (IVFPQ params)\n");
    printf("    [--train_size <int>] (how many base vectors used for training)\n");
    printf("    [--hits_csv <out_hits.csv>] (optional per-query topK dump)\n");
    printf("    [--hits_write_round 0/1] (default 0, 1 means dump for every round)\n");
    printf("    [--refine_r <int>] (default 1. r>1 enables exact rerank on top (k*r) candidates)\n");
    printf("    Note: nprobe_list is treated as nprobe_list for IVFPQ.\n");

    printf("Notes:");
    printf("- Scheme is fixed to A (true per-K timing).");
    printf("- Ground truth (GT) is computed ONCE with Kmax=max(k_list), then prefixes are reused for smaller K.");
    printf("- --k_list example: --k_list 1,5,10,20,50,100,200,500,1000");
    printf("- --nprobe_list example: --nprobe_list 2,4,8,16,32");
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
// skip_header: skip first line
// drop_first_col: ignore the first column (e.g., id)
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
 * Traditional recall@K (ANN-benchmarks style)
 *****************************************************/
static double recall_at_k_traditional_nq(
        const faiss::idx_t* I,      // [nq, I_stride]
        int I_stride,
        int k,
        const faiss::idx_t* gt,     // [nq, gt_stride]
        int gt_stride,
        int nq) {
    if (k <= 0 || nq <= 0) return 0.0;

    if (gt_stride < k || I_stride < k) {
        fprintf(stderr, "Invalid recall@%d: gt_stride=%d I_stride=%d\n", k, gt_stride, I_stride);
        return NAN;
    }

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
 * Exact rerank (in-memory "回表") for ONE query
 *
 * BUGFIXES:
 *  - avoid UB: nth_element(nth==end) is invalid
 *  - deduplicate candidate ids to avoid duplicate output ids
 *****************************************************/
static inline void refine_one_query_exact(
        const float* xb, size_t nb,
        const float* q, int d,
        const faiss::idx_t* Iref, int k_ref,
        faiss::MetricType mt,
        faiss::idx_t* Iout, float* Dout, int k) {

    // collect valid candidates
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

    // dedup candidates (prevents duplicate ids in final top-k)
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());

    struct Pair { float v; faiss::idx_t id; };
    std::vector<Pair> scored;
    scored.reserve(cand.size());

    if (mt == faiss::METRIC_L2) {
        // smaller is better
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

        // only nth_element when keep < size (nth must be in [begin,end))
        if (keep < (int)scored.size()) {
            std::nth_element(scored.begin(), scored.begin() + keep, scored.end(), cmp);
            scored.resize(keep);
        }
        std::sort(scored.begin(), scored.end(), cmp);

        int outn = std::min<int>(keep, k);
        for (int i = 0; i < outn; ++i) { Iout[i] = scored[i].id; Dout[i] = scored[i].v; }
        for (int i = outn; i < k; ++i) { Iout[i] = -1; Dout[i] = 0.0f; }
    } else {
        // METRIC_INNER_PRODUCT: larger is better
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

        // only nth_element when keep < size (nth must be in [begin,end))
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
 * Exact GT builder (IndexFlat)  ——only compute once at Kmax
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
    const std::string csv_out = get_arg_or(kv, "csv_path", "ivfpq_qps_recall_grid.csv");

    // per-query hits dump
    const std::string hits_csv = get_arg_or(kv, "hits_csv", "");
    const bool hits_write_round = (get_arg_or(kv, "hits_write_round", "0") == "1");

    // refine rerank factor
    const int refine_r = std::stoi(get_arg_or(kv, "refine_r", "1"));
    if (refine_r <= 0) {
        fprintf(stderr, "[FATAL] --refine_r must be >= 1\n");
        return 1;
    }

    // metric: l2 or ip
    std::string metric = get_arg_or(kv, "metric", "l2");
    if (metric != "l2" && metric != "ip") {
        fprintf(stderr, "[FATAL] --metric must be 'l2' or 'ip'\n");
        return 1;
    }

    faiss::MetricType mt = (metric == "ip") ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

    // Scheme A only
    const std::string scheme = "A";

    // K list
    std::vector<int> k_list = parse_int_list(get_arg_or(kv, "k_list", "5,10,50,100,200,500,1000"));
    if (k_list.empty()) {
        fprintf(stderr, "[FATAL] empty k_list\n");
        return 1;
    }
    int kmax = max_of(k_list);

    // nprobe list
    std::vector<int> efs = parse_int_list(get_arg_or(kv, "nprobe_list", "2,4,8,16,32"));
    if (efs.empty()) {
        fprintf(stderr, "[FATAL] empty nprobe_list\n");
        return 1;
    }

    // IVFPQ parameters
    int nlist = std::stoi(get_arg_or(kv, "nlist", "512"));
    int pq_m  = std::stoi(get_arg_or(kv, "m", "16"));
    int pq_nbits = std::stoi(get_arg_or(kv, "nbits", "8"));
    size_t train_size = (size_t)std::stoll(get_arg_or(kv, "train_size", "1000000000"));

    if (nlist <= 0 || pq_m <= 0 || pq_nbits <= 0) {
        fprintf(stderr, "[FATAL] --nlist/--m/--nbits must be positive\n");
        return 1;
    }

    /**********************
     * Data loading mode
     **********************/
    const std::string base_csv  = get_arg_or(kv, "base_csv", "");
    const std::string query_csv = get_arg_or(kv, "query_csv", "");
    const bool use_csv = (!base_csv.empty() && !query_csv.empty());

    if (!use_csv) {
        fprintf(stderr, "[FATAL] This program now only supports CSV inputs. Provide --base_csv and --query_csv.\n");
        print_usage(argv[0]);
        return 1;
    }

    // CSV options
    const bool csv_skip_header = (get_arg_or(kv, "csv_skip_header", "1") == "1");
    const bool csv_drop_first_col = (get_arg_or(kv, "csv_drop_first_col", "1") == "1");

    printf("=== Config ===\n");
    printf("mode      : CSV\n");
    printf("metric    : %s\n", metric.c_str());
    printf("scheme    : %s\n", scheme.c_str());
    printf("rounds    : %d\n", rounds);
    printf("csv_out   : %s\n", csv_out.c_str());
    printf("refine_r  : %d\n", refine_r);
    if (!hits_csv.empty()) {
        printf("hits_csv  : %s (write_each_round=%d)\n", hits_csv.c_str(), (int)hits_write_round);
    } else {
        printf("hits_csv  : (disabled)\n");
    }
    printf("k_list    : ");
    for (size_t i = 0; i < k_list.size(); i++) printf("%s%d", (i ? "," : ""), k_list[i]);
    printf("\n");
    printf("nprobe_list   : ");
    for (size_t i = 0; i < efs.size(); i++) printf("%s%d", (i ? "," : ""), efs[i]);
    printf("\n");
    printf("base_csv  : %s\n", base_csv.c_str());
    printf("query_csv : %s\n", query_csv.c_str());
    printf("csv_skip_header   : %d\n", (int)csv_skip_header);
    printf("csv_drop_first_col: %d\n", (int)csv_drop_first_col);
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

    if (d_base == 0 || d_query == 0 || d_base != d_query) {
        fprintf(stderr, "[FATAL] dim mismatch: base d=%zu, query d=%zu\n", d_base, d_query);
        return 1;
    }
    const int d = (int)d_base;

    printf("[DATA] nb=%zu d=%d, nq=%zu\n", nb, d, nq_all);

    if (nq_all > (size_t)INT_MAX) {
        fprintf(stderr, "[FATAL] nq=%zu exceeds INT_MAX; please split query_csv into smaller files.\n", nq_all);
        return 1;
    }

    // Use ALL queries in query_csv
    const size_t nq = nq_all;
    const int nq_i = (int)nq;
    float* xq = xq_all;
    xq_all = nullptr;

    /**********************
     * Ground Truth (GT only once at Kmax)
     **********************/
    const int ask_kmax = std::min<int>(kmax, (int)nb);

    faiss::idx_t* gt = nullptr;
    int gt_stride = 0;
    printf("[%.3f s] Building exact ground truth ONCE with IndexFlat (%s), Kmax=%d\n",
           now_sec() - t0, metric.c_str(), ask_kmax);
    gt = build_exact_gt(xb, nb, xq, nq, d, ask_kmax, metric);
    gt_stride = ask_kmax;
    printf("[GT] built shape: (%zu, %d)\n", nq, gt_stride);

    /**********************
     * Build FAISS IVFPQ once
     **********************/
    printf("\n==== Build IVFPQ once, then run (nprobe,K) grid searches (Scheme A only) ====\n");

    auto* quantizer = new faiss::IndexFlat(d, mt);
    auto* ivfpq = new faiss::IndexIVFPQ(quantizer, d, nlist, pq_m, pq_nbits, mt);
    ivfpq->verbose = true;
    ivfpq->own_fields = true;

    // Train IVFPQ (must train before add)
    size_t train_n = std::min(train_size, nb);
    if (train_n < (size_t)nlist) {
        fprintf(stderr, "[FATAL] train_n=%zu < nlist=%d; refuse to train.\n", train_n, nlist);
        return 1;
    }
    printf("[%.3f s] Training IVFPQ: train_n=%zu, nlist=%d, m=%d, nbits=%d\n",
           now_sec() - t0, train_n, nlist, pq_m, pq_nbits);
    double tb0 = now_sec();
    ivfpq->train((faiss::idx_t)train_n, xb);
    printf("[%.3f s] Training done in %.1fs\n", now_sec() - t0, now_sec() - tb0);

    printf("[%.3f s] Adding %zu base vectors\n",now_sec() - t0, nb);
    double tb1 = now_sec();
    ivfpq->add((faiss::idx_t)nb, xb);
    double tb2 = now_sec();
    printf("Build done in %.1fs\n", tb2 - tb1);

    /**********************
     * CSV output (summary)
     **********************/
    FILE* fout = fopen(csv_out.c_str(), "w");
    if (!fout) {
        perror("fopen csv");
        return 1;
    }
    fprintf(fout, "ef,k,scheme,qps_mean,qps_std,lat_ms_mean,lat_ms_std,sec_mean,sec_std,recall_mean,recall_std,rounds\n");
    fflush(fout);

    /**********************
     * hits CSV output (per-query topK)
     **********************/
    FILE* fhits = nullptr;
    if (!hits_csv.empty()) {
        fhits = fopen(hits_csv.c_str(), "w");
        if (!fhits) {
            perror("fopen hits_csv");
            return 1;
        }
        // renamed column: id_list (instead of layout_pos_list)
        fprintf(fhits, "nprobe,k,round,qi,id_list\n");
        fflush(fhits);
    }

    // Warmup
    {
        int warm_k = std::min<int>(10, (int)nb);
        std::vector<faiss::idx_t> Iw((size_t)nq * (size_t)warm_k);
        std::vector<float> Dw((size_t)nq * (size_t)warm_k);
        ivfpq->nprobe = efs.front();
        ivfpq->search((faiss::idx_t)nq, xq, warm_k, Dw.data(), Iw.data());
        printf("[%.3f s] Warmup done\n",now_sec() - t0);
    }

    /**********************
     * Scheme A only loop
     **********************/
    for (int ef : efs) {
        ivfpq->nprobe = ef;

        for (int k : k_list) {
            const int kk = std::min<int>(k, (int)nb);
            if (kk <= 0) continue;

            if (kk > ask_kmax) {
                fprintf(stderr, "[FATAL] internal: kk > ask_kmax\n");
                return 1;
            }
            if (kk > gt_stride) {
                fprintf(stderr,
                        "[FATAL] kk=%d > gt_stride=%d (GT not long enough). "
                        "Increase GT or reduce k_list.\n",
                        kk, gt_stride);
                return 1;
            }

            std::vector<double> secs, lats_ms, qpss, recs;
            secs.reserve((size_t)rounds);
            lats_ms.reserve((size_t)rounds);
            qpss.reserve((size_t)rounds);
            recs.reserve((size_t)rounds);

            // final outputs (after optional refine)
            std::vector<faiss::idx_t> I((size_t)nq * (size_t)kk);
            std::vector<float>        D((size_t)nq * (size_t)kk);

            // coarse outputs (k_ref candidates)
            const int k_ref = std::min<int>((int)nb, kk * refine_r);
            std::vector<faiss::idx_t> Iref((size_t)nq * (size_t)k_ref);
            std::vector<float>        Dref((size_t)nq * (size_t)k_ref);

            for (int rr = 0; rr < rounds; rr++) {
                double t1 = now_sec();

                // 1) coarse search returns k_ref candidates (PQ distance)
                ivfpq->search((faiss::idx_t)nq, xq, k_ref, Dref.data(), Iref.data());

                // 2) exact refine (回表) on candidates -> final top kk
                if (refine_r > 1) {
                    for (int qi = 0; qi < nq_i; ++qi) {
                        const float* qvec = xq + (size_t)qi * (size_t)d;
                        const faiss::idx_t* Iq_ref = Iref.data() + (size_t)qi * (size_t)k_ref;
                        faiss::idx_t* Iq_out = I.data() + (size_t)qi * (size_t)kk;
                        float* Dq_out = D.data() + (size_t)qi * (size_t)kk;

                        refine_one_query_exact(xb, nb, qvec, d, Iq_ref, k_ref, mt, Iq_out, Dq_out, kk);
                    }
                } else {
                    // no refine: take first kk from coarse results
                    for (int qi = 0; qi < nq_i; ++qi) {
                        const faiss::idx_t* srcI = Iref.data() + (size_t)qi * (size_t)k_ref;
                        const float* srcD = Dref.data() + (size_t)qi * (size_t)k_ref;
                        faiss::idx_t* dstI = I.data() + (size_t)qi * (size_t)kk;
                        float* dstD = D.data() + (size_t)qi * (size_t)kk;
                        std::memcpy(dstI, srcI, (size_t)kk * sizeof(faiss::idx_t));
                        std::memcpy(dstD, srcD, (size_t)kk * sizeof(float));
                    }
                }

                double t2 = now_sec();

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

                // dump per-query topK to hits_csv (uses FINAL I -> refined results)
                if (fhits && (hits_write_round || rr == 0)) {
                    for (int qi = 0; qi < nq_i; ++qi) {
                        const faiss::idx_t* Ii = I.data() + (size_t)qi * (size_t)kk;

                        // 拼接 id_list，用“空格”分隔
                        std::string s;
                        s.reserve((size_t)kk * 12);

                        for (int r = 0; r < kk; ++r) {
                            if (r) s.push_back(' ');
                            s += std::to_string((long long)Ii[r]);
                        }

                        fprintf(fhits, "%d,%d,%d,%d,\"%s\"\n",
                                ef, kk, rr, qi, s.c_str());
                    }
                    fflush(fhits);
                }
            }

            double sec_mean, sec_std, qps_mean, qps_std, lat_mean, lat_std, rec_mean, rec_std;
            mean_std(secs, &sec_mean, &sec_std);
            mean_std(qpss, &qps_mean, &qps_std);
            mean_std(lats_ms, &lat_mean, &lat_std);
            mean_std(recs, &rec_mean, &rec_std);

            fprintf(fout,
                    "%d,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
                    ef, kk, "A",
                    qps_mean, qps_std,
                    lat_mean, lat_std,
                    sec_mean, sec_std,
                    rec_mean, rec_std,
                    rounds);
            fflush(fout);

            printf("nprobe=%d k=%d | refine_r=%d | QPS=%.1f | lat=%.3f ms/q | R@%d=%.4f\n",
                   ef, kk, refine_r, qps_mean, lat_mean, kk, rec_mean);
        }
    }

    fclose(fout);
    if (fhits) fclose(fhits);

    // cleanup
    delete ivfpq;
    delete[] xb;
    delete[] xq;
    delete[] gt;

    printf("\n[%.3f s] Done. CSV written to %s\n",now_sec() - t0, csv_out.c_str());
    if (!hits_csv.empty()) {
        printf("[%.3f s] Hits CSV written to %s\n", now_sec() - t0, hits_csv.c_str());
    }
    return 0;
}



