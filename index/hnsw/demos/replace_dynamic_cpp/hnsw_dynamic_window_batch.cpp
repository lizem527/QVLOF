//
// Created by 许晶精 on 1/23/26.
//

/**
 * HNSW query performance benchmark (Scheme A only):
 *  - Scheme A: run a real search() for each (ef, K) to get true QPS@K and Recall@K
 *
 * Supports loading vectors from:
 *  - CSV (id,v0..v127) or plain v0..v127
 *
 * Ground truth:
 *  - computed ONCE with Kmax=max(k_list), then prefixes reused for smaller K.
 *
 * Output CSV columns:
 * ef,k,scheme,qps_mean,qps_std,lat_ms_mean,lat_ms_std,sec_mean,sec_std,recall_mean,recall_std,rounds
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

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>   // <-- changed (was IndexMIRAGE.h)

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
    printf("    [--rounds <int>] [--k_list <csv>] [--ef_list <csv>] [--csv_path <out.csv>]\n");
    printf("    [--ids_csv_path <out_ids.csv>] [--ids_round_mode first|all]\n");
    printf("    [--hnsw_M <int>] [--hnsw_efC <int>]\n");

    printf("Notes:\n");
    printf("- Scheme is fixed to A (true per-K timing).\n");
    printf("- Ground truth (GT) is computed ONCE with Kmax=max(k_list), then prefixes are reused.\n");
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
        fprintf(stderr, "CSV empty: %s\n", path);
        exit(1);
    }

    float* out = new float[rows * dim];
    std::memcpy(out, data.data(), rows * dim * sizeof(float));
    *n0 = rows;
    *n1 = dim;
    return out;
}

/*****************************************************
 * Stats helpers
 *****************************************************/
static void mean_std(const std::vector<double>& v, double* mean, double* stdev) {
    if (v.empty()) { *mean = 0; *stdev = 0; return; }
    double s = 0;
    for (double x : v) s += x;
    double m = s / (double)v.size();
    double ss = 0;
    for (double x : v) {
        double d = x - m;
        ss += d * d;
    }
    double var = (v.size() > 1) ? (ss / (double)(v.size() - 1)) : 0.0;
    *mean = m;
    *stdev = std::sqrt(var);
}

/*****************************************************
 * Traditional recall@K
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
        return NAN; // 或直接 abort/assert
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


/*****************************************************
 * Build exact GT once with IndexFlat
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





// ----------------------------
// Per-query id_list dump (layout ids)
// ----------------------------
struct IdsDumpConfig {
    std::string path;
    bool enabled = false;
    bool round_all = false;             // false => first round only
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
    // NEW format
    fprintf(f, "ef,k,round,query,id_list\n");
    fflush(f);
    return f;
}






// NEW: one row per query: ef,k,round,query,"id0 id1 ... id(k-1)"
static void dump_ids_one_round_rowwise(
        FILE* fids,
        int ef, int kk, int rr,
        const faiss::idx_t* I,
        int nq) {
    if (!fids) return;

    for (int qi = 0; qi < nq; qi++) {
        const faiss::idx_t* ip = I + (size_t)qi * (size_t)kk;

        // header columns
        fprintf(fids, "%d,%d,%d,%d,\"", ef, kk, rr, qi);

        // id list (space-separated inside quotes)
        for (int r = 0; r < kk; r++) {
            faiss::idx_t id = ip[r];
            if (r) fputc(' ', fids);
            fprintf(fids, "%ld", (long)id);
        }

        fprintf(fids, "\"\n");
    }
    fflush(fids);
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




/* add for dynamic query */
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




int main(int argc, char** argv) {
    double t0 = now_sec();
    auto args = parse_kv_args(argc, argv);
    if (args.count("help")) {
        print_usage(argv[0]);
        return 0;
    }
    // Per-query id_list dump (layout ids)
    IdsDumpConfig ids_cfg = parse_ids_dump_config(args);
    FILE* fids = open_ids_csv_if_enabled(ids_cfg);



    const std::string base_csv = get_arg_or(args, "base_csv", "");
    const std::string query_csv = get_arg_or(args, "query_csv", "");
    const std::string metric   = get_arg_or(args, "metric", "l2");

    const bool csv_skip_header   = (std::stoi(get_arg_or(args, "csv_skip_header", "1")) != 0);
    const bool csv_drop_first_col= (std::stoi(get_arg_or(args, "csv_drop_first_col", "1")) != 0);

    const int rounds = std::stoi(get_arg_or(args, "rounds", "5"));
    const std::string k_list_s  = get_arg_or(args, "k_list", "5,10,100,200,500,1000");
    const std::string ef_list_s = get_arg_or(args, "ef_list", "20,40,100,200");
    const std::string csv_out   = get_arg_or(args, "csv_path", "mirage_out.csv");

    // HNSW params (new)
    const int hnsw_M  = std::stoi(get_arg_or(args, "hnsw_M",  "32"));
    const int hnsw_efC= std::stoi(get_arg_or(args, "hnsw_efC","200"));

    /* add for dynamic query */
    const bool do_dynamic = (std::stoi(get_arg_or(args, "dynamic", "0")) != 0);
    const int window_size = std::stoi(get_arg_or(args, "window_size", "200"));
    const int dynamic_k = std::stoi(get_arg_or(args, "dynamic_k", "20"));
    const std::string per_query_path = get_arg_or(args, "per_query_csv_path", "");
    const std::string per_window_path = get_arg_or(args, "per_window_csv_path", "");



    if (base_csv.empty() || query_csv.empty()) {
        fprintf(stderr, "[FATAL] --base_csv and --query_csv are required.\n");
        print_usage(argv[0]);
        return 1;
    }

    std::vector<int> k_list = parse_int_list(k_list_s);
    std::vector<int> efs    = parse_int_list(ef_list_s);
    if (k_list.empty() || efs.empty()) {
        fprintf(stderr, "[FATAL] k_list or ef_list is empty.\n");
        return 1;
    }
    const int kmax = max_of(k_list);

    // Load CSV
    size_t nb = 0, d_base = 0;
    float* xb = nullptr;

    size_t nq_all = 0, d_query = 0;
    float* xq_all = nullptr;

    printf("[%.3f s] Loading base vectors (CSV)\n", now_sec() - t0);
    xb = csv_read_f32_2d(base_csv.c_str(), &nb, &d_base, csv_skip_header, csv_drop_first_col);

    printf("[%.3f s] Loading queries (CSV)\n", now_sec() - t0);
    xq_all = csv_read_f32_2d(query_csv.c_str(), &nq_all, &d_query, csv_skip_header, csv_drop_first_col);

    if (d_base == 0 || d_query == 0 || d_base != d_query) {
        fprintf(stderr, "[FATAL] dim mismatch: base d=%zu, query d=%zu\n", d_base, d_query);
        return 1;
    }
    const int d = (int)d_base;

    printf("[DATA] nb=%zu d=%d, nq=%zu\n", nb, d, nq_all);

    if (nq_all > (size_t)INT_MAX) {
        fprintf(stderr, "[FATAL] nq=%zu exceeds INT_MAX; please split query_csv.\n", nq_all);
        return 1;
    }

    // Use ALL queries
    const size_t nq = nq_all;
    const int nq_i = (int)nq;
    float* xq = xq_all;
    xq_all = nullptr;

    /**********************
     * Ground Truth (GT once at Kmax)
     **********************/
    const int ask_kmax = std::min<int>(kmax, (int)nb);

    printf("[%.3f s] Building exact ground truth ONCE with IndexFlat (%s), Kmax=%d\n",
           now_sec() - t0, metric.c_str(), ask_kmax);

    faiss::idx_t* gt = build_exact_gt(xb, nb, xq, nq, d, ask_kmax, metric);
    const int gt_stride = ask_kmax;
    printf("[GT] built shape: (%zu, %d)\n", nq, gt_stride);

    /**********************
     * Build HNSW once
     **********************/
    printf("\n==== Build HNSW once, then run (ef,K) grid searches (Scheme A only) ====\n");

    faiss::MetricType mt = faiss::METRIC_L2;
    if (metric == "ip") mt = faiss::METRIC_INNER_PRODUCT;

    auto* hnsw = new faiss::IndexHNSWFlat(d, hnsw_M, mt);
    hnsw->verbose = true;
    hnsw->hnsw.efConstruction = hnsw_efC;

    printf("[%.3f s] Adding %zu base vectors\n", now_sec() - t0, nb);
    double tb1 = now_sec();
    hnsw->add((faiss::idx_t)nb, xb);
    double tb2 = now_sec();
    printf("Build done in %.1fs\n", tb2 - tb1);

    /**********************
     * CSV output
     **********************/



    /**********************
     * Scheme A only loop
     **********************/


    if (do_dynamic) {
        printf("\n==== Dynamic window-level benchmark ====\n");

        const int kk = std::min(dynamic_k, (int)nb);
        if (kk > gt_stride) {
            fprintf(stderr, "[FATAL] dynamic_k > gt_stride\n");
            exit(1);
        }

        const int nwin = (nq_i + window_size - 1) / window_size;

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

        }

        // warm up
        {
            int warm_k = std::min(kk, (int)nb);
            int wn = std::min(window_size, nq_i);
            std::vector<faiss::idx_t> Iw((size_t)wn * (size_t)warm_k);
            std::vector<float> Dw((size_t)wn * (size_t)warm_k);
            hnsw->hnsw.efSearch = efs.front();
            hnsw->search((faiss::idx_t)wn, xq, warm_k, Dw.data(), Iw.data());
        }


        for (int ef : efs) {
            hnsw->hnsw.efSearch = ef;

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
                double rec_std = 0.0;


                for (int rr = 0; rr < rounds; rr++){
                    double t1 = now_sec();
                    hnsw->search(
                            (faiss::idx_t)wn,
                            xq + (size_t)qb * (size_t)d,
                            kk,
                            Dw.data(),
                            Iw.data()
                    );
                    double t2 = now_sec();


                    double sec = t2 - t1;
                    double lat_ms = (sec > 0) ? (sec * 1000.0 / (double)wn) : 0.0;

                    // dump top-k ids per query (OPTIONAL)
                    if (fids) {
                        bool do_dump = ids_cfg.round_all ? true : (rr == 0);
                        if (do_dump) {
                            dump_ids_one_round_rowwise_with_qbase(fids, ef, kk, rr, Iw.data(), wn, qb);
                        }
                    }



                    secs_w.push_back(sec);
                    lats_w.push_back(lat_ms);

                    if(rr == 0){
                        double rec_sum = 0.0;
                        for (int i = 0; i < wn; i++) {
                            const faiss::idx_t* Ii =
                                    Iw.data() + (size_t)i * (size_t)kk;
                            const faiss::idx_t* gti =
                                    gt + (size_t)(qb + i) * (size_t)gt_stride;

                            rec_sum += recall_one_query(Ii, kk, gti, gt_stride);
                        }
                        rec_mean = rec_sum / (double)wn;
                        rec_std  = 0.0; // 单次测量，std 设为 0（或写 NAN 也行）

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
                }

                printf("ef=%d window=%d | R@%d=%.4f±%.4f | lat=%.3f±%.3f ms/q\n",
                       ef, w, kk, rec_mean, rec_std, lat_mean, lat_std);
            }

            if(fpw){
                fflush(fpw);
            }

        }

        if (fpw) fclose(fpw);
    }

    if (fids) fclose(fids);

    // cleanup
    delete hnsw;
    delete[] xb;
    delete[] xq;
    delete[] gt;

    printf("\n[%.3f s] Done. CSV written to %s\n", now_sec() - t0, csv_out.c_str());
    return 0;
}


