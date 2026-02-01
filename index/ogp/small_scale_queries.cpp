//
// Created by 许晶精 on 1/26/26.
//

//
// Created by 许晶精 on 1/20/26.
//

//
// Created by 许晶精 on 1/15/26.
//

#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>

#include <parlay/primitives.h>

#include "hnsw_router.h"
#include "inverted_index.h"
#include "kmeans_tree_router.h"
#include "metis_io.h"
#include "points_io.h"
#include "recall.h"

#include "inverted_index_hnsw.h"


void DedupNeighbors(std::vector<NNVec>& neighbors, size_t num_neighbors) {
    parlay::parallel_for(0, neighbors.size(), [&](size_t i) {
        auto& n = neighbors[i];
        // 1) dedup by global id
        std::sort(n.begin(), n.end(), [](const auto& l, const auto& r) { return l.second < r.second; });
        n.erase(std::unique(n.begin(), n.end()), n.end());
        // 2) sort by (dis, id) and keep topK
        std::sort(n.begin(), n.end());
        n.resize(std::min(n.size(), num_neighbors));
    });
}

/* helper func: 把 neighs[i].second ，即 global id 串成一个字符串 */
static inline std::string JoinIdList(const NNVec& neighs){
    // neighs[i].second is global id (the one used in ground truth)
    std::ostringstream oss;
    for(size_t i = 0; i < neighs.size(); ++i){
        if(i){
            oss << ' ';
        }
        oss << neighs[i].second;
    }
    return oss.str();
}



/* add for dynamic query */
static inline void DedupOne(NNVec& n, size_t num_neighbors) {
    // NNVec = vector<pair<float /*dist*/, uint32_t /*id*/>>

    // 1) 按 (id, dist) 排序：同 id 时距离更小的在前
    std::sort(n.begin(), n.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second < b.second; // id
        return a.first < b.first;                             // dist
    });

    // 2) 按 id 去重：保留每个 id 的第一条（即最小 dist）
    auto last = std::unique(n.begin(), n.end(), [](const auto& a, const auto& b) {
        return a.second == b.second; // same id
    });
    n.erase(last, n.end());

    // 3) 再按距离升序排序（RecallOne 需要 top-k 按距离递增）
    std::sort(n.begin(), n.end(), [](const auto& a, const auto& b) {
        return a.first < b.first; // dist
    });

    // 4) 截断 top-k
    if (n.size() > num_neighbors) n.resize(num_neighbors);
}

// 单条 query 的 recall@k：距离 <= GT 第k邻居距离的命中比例
static inline double RecallOne(const NNVec& neighs, float kth_dist, int k) {
    int hit = 0;
    int upto = std::min<int>((int)neighs.size(), k);
    for (int i = 0; i < upto; i++) {
        if (neighs[i].first <= kth_dist) hit++;  // first 是 dist
    }
    return double(hit) / double(k);
}

struct WindowStats {
    size_t cnt = 0;
    double sum_lat_ms = 0.0;
    double sum_routing_ms = 0.0;
    double sum_query_ms = 0.0;
    double sum_recall = 0.0;

    void add(double lat_ms, double routing_ms, double query_ms, double recall) {
        cnt++;
        sum_lat_ms += lat_ms;
        sum_routing_ms += routing_ms;
        sum_query_ms += query_ms;
        sum_recall += recall;
    }
};




int main(int argc, const char* argv[]) {
    if (argc != 8) {
        std::cerr << "Usage ./SmallScaleQueries input-points queries ground-truth-file num-neighbors partition part-method out-file" << std::endl;
        std::abort();
    }

    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string = argv[4];
    int num_neighbors = std::stoi(k_string);
    std::string partition_file = argv[5];
    std::string part_method = argv[6];
    std::string out_file = argv[7];

    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);

    std::vector<NNVec> ground_truth;
    if (std::filesystem::exists(ground_truth_file)) {
        ground_truth = ReadGroundTruth(ground_truth_file);
    } else {
        std::cout << "start computing ground truth" << std::endl;
        ground_truth = ComputeGroundTruth(points, queries, num_neighbors);
        std::cout << "computed ground truth" << std::endl;
        WriteGroundTruth(ground_truth_file, ground_truth);
        std::cout << "wrote ground truth to file " << ground_truth_file << std::endl;
    }
    std::vector<float> distance_to_kth_neighbor = ConvertGroundTruthToDistanceToKthNeighbor(ground_truth, num_neighbors, points, queries);
    std::cout << "finished converting ground truth to distances" << std::endl;

    if (!std::filesystem::exists(partition_file) || part_method == "None") {
        std::cout << "Not partitioned. --> Run HNSW directly on input" << std::endl;
        Timer timer;
        timer.Start();
        HNSWParameters hnsw_parameters;
#ifdef MIPS_DISTANCE
        hnswlib::InnerProductSpace space(points.d);
#else
        hnswlib::L2Space space(points.d);
#endif
        hnswlib::HierarchicalNSW<float> hnsw(&space, points.n, hnsw_parameters.M, hnsw_parameters.ef_construction, 555);
        parlay::parallel_for(0, points.n, [&](size_t i) { hnsw.addPoint(points.GetPoint(i), i); });
        std::cout << "Building HNSW took " << timer.Stop() << " seconds." << std::endl;

        for (int ef : { 20, 50, 80, 100, 120, 150, 200, 300, 400 }) {
            std::vector<NNVec> neighbors(queries.n);
            hnsw.setEf(ef);
            timer.Start();
            for (size_t q = 0; q < queries.n; ++q) {
                auto result_pq = hnsw.searchKnn(queries.GetPoint(q), num_neighbors);
                NNVec result;
                while (!result_pq.empty()) {
                    result.emplace_back(result_pq.top());
                    result_pq.pop();
                }
                neighbors[q] = std::move(result);
            }
            double time = timer.Stop();
            double recall = Recall(neighbors, distance_to_kth_neighbor, num_neighbors);
            std::cout << "HNSW query with ef = " << ef << " took " << time << " seconds. recall = " << recall << ". avg latency = " << 1000.0 * time / queries.n
                      << " ms."
                      << " avg dist comps " << static_cast<double>(hnsw.metric_distance_computations) / queries.n << std::endl;
        }

        return 0;
    }

    Clusters clusters = ReadClusters(partition_file);
    int num_shards = clusters.size();

    Timer timer;
    timer.Start();
    KMeansTreeRouterOptions options{ .num_centroids = 32, .min_cluster_size = 200, .budget = 20000, .search_budget = 200 };
    KMeansTreeRouter router;
    router.Train(points, clusters, options);
    std::cout << "Training KMTR took " << timer.Stop() << " seconds." << std::endl;

    auto [routing_points, routing_index_partition] = router.ExtractPoints();



    //std::vector<std::tuple<std::string /*router*/, double /*routing time*/, std::vector<std::vector<int>> /*probes*/>> probes_v;
    std::vector<std::tuple<
            std::string,                 // router name
            std::vector<double>,         // routing_ms_per_q
            std::vector<std::vector<int>>// probes
    >> probes_v;



    /*
    std::vector<std::vector<int>> buckets_to_probe_kmtr(queries.n);
    timer.Start();
    for (size_t q = 0; q < queries.n; ++q) {
        buckets_to_probe_kmtr[q] = router.Query(queries.GetPoint(q), options.search_budget);
    }
    double time = timer.Stop();
    std::cout << "KMTR routing took " << time << " seconds. That's " << 1000.0 * time / queries.n << "ms per query, or " << queries.n / time << " QPS"
              << std::endl;
    probes_v.emplace_back(std::tuple("KMTR", time, std::move(buckets_to_probe_kmtr)));
     */

    std::vector<std::vector<int>> buckets_to_probe_kmtr(queries.n);
    std::vector<double> routing_ms_per_q_kmtr(queries.n, 0.0);

    Timer t_route_total;
    Timer t_route_one;

    t_route_total.Start();
    for (size_t q = 0; q < queries.n; ++q) {
        t_route_one.Start();
        buckets_to_probe_kmtr[q] = router.Query(queries.GetPoint(q), options.search_budget);
        routing_ms_per_q_kmtr[q] = 1000.0 * t_route_one.Stop();
    }
    double routing_time = t_route_total.Stop();

    std::cout << "KMTR routing took " << routing_time << " seconds. That's "
              << 1000.0 * routing_time / queries.n << "ms per query, or "
              << queries.n / routing_time << " QPS" << std::endl;

    probes_v.emplace_back(std::tuple(
            "KMTR",
            std::move(routing_ms_per_q_kmtr),
            std::move(buckets_to_probe_kmtr)
    ));







    timer.Start();
    InvertedIndexHNSW ivf_hnsw(points);
    ivf_hnsw.hnsw_parameters = HNSWParameters{ .M = 16, .ef_construction = 32, .ef_search = 256 }; // ef_construction 从 200 改成了 32
    ivf_hnsw.Build(points, clusters);
    std::cout << "Building IVF-HNSW took " << timer.Restart() << " seconds." << std::endl;
    InvertedIndex ivf(points, clusters);
    std::cout << "Building IVF took " << timer.Stop() << " seconds." << std::endl;

    std::cout << "Finished building IVFs" << std::endl;

    std::ofstream out(out_file);
    out << "partitioning,routing,shard query,probes,latency,routing latency, query latency,recall" << std::endl;


    /* 开始 query 处理 */
    // 输出 2 个文件：per-query 和 per-window (基于 out_file 做前缀)
    std::string per_query_path = out_file + ".per_query.csv";
    std::string per_window_path = out_file + ".per_window.csv";

    std::ofstream out_q(per_query_path);
    std::ofstream out_w(per_window_path);
    out_q << "partitioning,routing,shard_query,num_probes,probed_shard,window_id,query_id,total_lat_ms,routing_lat_ms,query_lat_ms,shard_lat_ms, recall, merged_id_list\n";
    out_w << "partitioning,routing,shard_query,num_probes,window_id,num_queries,avg_total_lat_ms,avg_routing_lat_ms,avg_query_lat_ms,avg_recall\n";


    std::cout << "Start queries" << std::endl;

    const int window_size = 200;
    const int n_windows = (int)( (queries.n + window_size - 1) / window_size);


    for (const auto& [desc, routing_ms_per_q, probes] : probes_v) {
        //routing_time 是对全量 queries 计时出来的，所以这里按 query 均摊(后续可加上 per-window routing )
        //const double routing_lat_ms = 1000.0 * (routing_time / queries.n);


        std::vector<NNVec> neighbors(queries.n);
        std::vector<double> cum_query_ms(queries.n, 0.0);
        //time = 0;
        for (int num_probes = 1; num_probes <= num_shards; ++num_probes) {

            std::vector<WindowStats> wstats(n_windows);

            //timer.Start();
            for (size_t q = 0; q < queries.n; ++q) {

                int window_id = (int) q / window_size;


                int shard = probes[q][num_probes-1];
                timer.Start();
                //auto neighs = ivf_hnsw.QueryBucket(queries.GetPoint(q), num_neighbors, probes[q][num_probes - 1]);
                auto neighs = ivf_hnsw.QueryBucket(queries.GetPoint(q), num_neighbors, shard);
                double q_ms = 1000.0 * timer.Stop();

                // shard top-k id list
                NNVec shard_neighs = neighs;
                DedupOne(shard_neighs, num_neighbors);
                std::string shard_ids = JoinIdList(shard_neighs);

                // merged top-k id list (cumulative)
                neighbors[q].insert(neighbors[q].end(), neighs.begin(), neighs.end());
                DedupOne(neighbors[q], num_neighbors);
                double r = RecallOne(neighbors[q], distance_to_kth_neighbor[q], num_neighbors);
                // id list
                std::string merged_ids = JoinIdList(neighbors[q]);

                cum_query_ms[q] += q_ms;

                double query_lat_ms = cum_query_ms[q];
                //double total_lat_ms = routing_lat_ms + query_lat_ms;
                double routing_lat_ms = routing_ms_per_q[q];
                double total_lat_ms = routing_lat_ms + query_lat_ms;

                out_q << part_method << "," << desc << ","
                      << "HNSW" << ","
                      << num_probes << ","
                      << shard << ","
                      << window_id << ","
                      << q << ","
                      << total_lat_ms << ","
                      << routing_lat_ms << ","
                      << query_lat_ms << ","
                      << q_ms << ","
                      << r << ","
                      << "\"" << merged_ids << "\""
                      << "\n";

                wstats[window_id].add(total_lat_ms, routing_lat_ms, query_lat_ms, r);
            }

            for(int w = 0; w < n_windows; ++w){
                const auto& s = wstats[w];
                if(s.cnt == 0){
                    continue;
                }
                out_w << part_method << "," << desc << ","
                      << "HNSW" << ","
                      << num_probes << ","
                      << w << ","
                      << s.cnt << ","
                      << (s.sum_lat_ms / s.cnt) << ","
                      << (s.sum_routing_ms / s.cnt) << ","
                      << (s.sum_query_ms / s.cnt) << ","
                      << (s.sum_recall / s.cnt) << "\n";
            }

            std::cout << "[per-query] router=" << desc << " query=IVF-HNSW nprobes=" << num_probes << " done.\n";

        }

    }

    std::cout << "Wrote per-query to: " << per_query_path << "\n";
    std::cout << "Wrote per-window to: " << per_window_path << "\n";


}



















