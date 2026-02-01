#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <time.h>

#include "Preprocess.h"
#include "divGraph.h"
#include "fastGraph.h"
#include "Query.h"
#include "basis.h"

#if defined(unix) || defined(__unix__)
struct llt
{
    int date, h, m, s;
    llt(size_t diff) { set(diff); }
    void set(size_t diff)
    {
        date = diff / 86400;
        diff = diff % 86400;
        h = diff / 3600;
        diff = diff % 3600;
        m = diff / 60;
        s = diff % 60;
    }
};
#endif

// 把 q->res 的前 topk 个 id 拼成 "id1 id2 id3 ..."（id == base 下标）
static inline std::string build_id_list_str(const queryN* q, int topk) {
    std::ostringstream oss;
    int m = std::min<int>(topk, (int)q->res.size());
    for (int i = 0; i < m; ++i) {
        if (i) oss << ' ';
        oss << q->res[i].id;
    }
    return oss.str();
}

// 方案 2：vector + sort + 双指针，计算交集个数
// 不改变口径：与 GT 的前 num0 个（num0=min(res.size, k, gt_cap)）求交集，最后除以 k
static inline unsigned count_intersection_sorted(std::vector<unsigned>& a,
                                                 std::vector<unsigned>& b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    unsigned i = 0, j = 0, cnt = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) {
            ++i;
        } else if (b[j] < a[i]) {
            ++j;
        } else {
            ++cnt;
            ++i;
            ++j;
        }
    }
    return cnt;
}

template <class Graph>
void graphSearch(float c, int k, Graph* myGraph, Preprocess& prep, float beta,
                 std::string& datasetName, std::string& data_fold, int qType,
                 const char* tag)
 {
    if (!myGraph) return;

    // 跑满 bench 里的 query 数
    int Qnum = (int)prep.benchmark.N;
    if (Qnum <= 0) {
        std::cerr << "[ERR] benchmark.N <= 0, cannot run queries\n";
        return;
    }

    // 确保输出目录存在：data_fold + "ANN/"
    std::string ann_dir = data_fold + "ANN/";
    if (!GenericTool::CheckPathExistence(ann_dir.c_str())) {
        GenericTool::EnsurePathExistence(ann_dir.c_str());
    }

    std::string perq_path = ann_dir + datasetName + ".per_query.csv";

    // 如果是空文件，写表头
    {
        std::ifstream fin(perq_path);
        bool need_header = (!fin.good() || fin.peek() == std::ifstream::traits_type::eof());
        if (need_header) {
            std::ofstream fout(perq_path, std::ios::app);
            fout << "dataset,tag,k,ef,qid,time_ms,qps,recall,id_list\n";
        }
    }

    std::ofstream perq(perq_path, std::ios::app);

    Performance perform;

    // A 方案：per-query 先写到内存 buffer，最后一次性落盘（减少 I/O 干扰，利于显化 layout）
    std::ostringstream perq_buf;

    // 逐 query 搜索
    for (int j = 0; j < Qnum; j++) {
        queryN* q = new queryN((unsigned)j, c, (unsigned)k, prep, beta);

        switch (qType % 2) {
            case 0: myGraph->knn(q);     break;
            case 1: myGraph->knnHNSW(q); break;
            default: myGraph->knn(q);    break;
        }

        // ---- per-query recall（方案2：vector+sort+双指针，口径不变）----
        unsigned gt_cap = (unsigned)prep.benchmark.num;
        unsigned num0 = (unsigned)q->res.size();
        if (num0 > (unsigned)q->k) num0 = (unsigned)q->k;
        if (num0 > gt_cap) num0 = gt_cap;

        std::vector<unsigned> res_ids;
        std::vector<unsigned> gt_ids;
        res_ids.reserve(num0);
        gt_ids.reserve(num0);

        for (unsigned t = 0; t < num0; t++) {
            res_ids.push_back((unsigned)q->res[t].id);                 // base 下标
            gt_ids.push_back((unsigned)prep.benchmark.indice[q->flag][t]); // GT 前缀
        }

        unsigned hits = count_intersection_sorted(res_ids, gt_ids);
        float recall = (k > 0) ? ((float)hits / (float)k) : 0.0f;

        // timeTotal 在工程里是秒
        float time_s  = q->timeTotal;
        float time_ms = time_s * 1000.0f;
        float qps     = (time_s > 0) ? (1.0f / time_s) : 0.0f;

        // Top-k id_list（base 下标列表）
        std::string id_list = build_id_list_str(q, k);

        // 写到内存 buffer
        perq_buf << datasetName << "," << (tag ? tag : "") << ","
                 << k << "," << myGraph->ef << "," << q->flag << ","
                 << time_ms << "," << qps << "," << recall << ","
                 << "\"" << id_list << "\""
                 << "\n";


        // overall 统计
        perform.update(q, prep);

        delete q;
    }

    // 一次性落盘
    perq << perq_buf.str();
    perq.close();

    std::string algName, qt;
    switch (qType/2) {
        case 0:
            algName = "fastG";
            break;
        case 1:
            algName = "divGraph";
            break;
    }

    switch (qType % 2) {
        case 0:
            qt = "Fast";
            break;
        case 1:
            qt = "HNSW";
            break;
    }

    float mean_time = (float)perform.timeTotal / perform.num;
    float cost = ((float)perform.cost) / ((float)perform.num);
    float ratio = ((float)perform.prunings) / (perform.cost);
    float cost_total = myGraph->S + cost / (1 - ratio) * (((float)myGraph->lowDim) / myGraph->dim) + cost;
    float cpq = myGraph->L * myGraph->K + _lsh_UB + cost / (1 - ratio) * (((float)myGraph->lowDim) / myGraph->dim);

    std::stringstream ss;
    ss << std::setw(_lspace) << algName
       << std::setw(_sspace) << k
       << std::setw(_sspace) << myGraph->ef
       << std::setw(_lspace) << mean_time * 1000
       << std::setw(_lspace) << ((float)perform.NN_num) / (perform.num * k)
       << std::setw(_lspace) << ((float)perform.cost) / ((float)perform.num)
       << std::setw(_lspace) << cpq
       << std::setw(_lspace) << cost_total
       << std::setw(_lspace) << ((float)perform.prunings) / (perform.cost)
       << std::endl;

    time_t now = time(0);
    time_t zero_point = 1635153971; // 2021.10.25 17:27
    std::cout << ss.str();

    std::string query_result(myGraph->getFilename());
    auto idx = query_result.rfind('/');
    if( idx != std::string::npos){
        query_result.resize(idx + 1);
    }else{
        query_result.clear();
    }
    //query_result.assign(query_result.begin(), query_result.begin() + idx + 1);
    query_result += "result.txt";
    std::ofstream osf(query_result, std::ios_base::app);
    osf.seekp(0, std::ios_base::end);
    osf << ss.str();
    osf.close();

    float date = ((float)(now - zero_point)) / 86400;

    std::string fpath = data_fold + "ANN/";
    if (!GenericTool::CheckPathExistence(fpath.c_str())) {
        GenericTool::EnsurePathExistence(fpath.c_str());
        std::cout << BOLDGREEN << "WARNING:\n" << GREEN
                  << "Could not find the path of result file. Have created it. \n"
                  << "The query result will be stored in: " << fpath.c_str() << RESET;
    }

    std::ofstream os(fpath + "LSH-G_div_result.csv", std::ios_base::app);
    if (os) {
        os.seekp(0, std::ios_base::end);
        int tmp = (int)os.tellp();
        if (tmp == 0) {
            os << "Dataset,tage,k,L,K,T,RATIO,RECALL,AVG_TIME,COST,DATE" << std::endl;
        }
        std::string dataset = datasetName;
        os << dataset << ',' << (tag ? tag : "") << ','
           << k << ',' << myGraph->L << ',' << myGraph->K << ','
           << myGraph->T << ','
           << ((float)perform.ratio) / (perform.resNum) << ','
           << ((float)perform.NN_num) / (perform.num * k) << ','
           << mean_time * 1000 << ','
           << ((float)perform.cost) / (perform.num * prep.data.N) << ','
           << date << ','
           << std::endl;
        os.close();
    }
}

void zlshKnn(float c, int k, e2lsh& myLsh, Preprocess& prep, float beta, std::string& datasetName, std::string& data_fold) {

    int T = 10;

    Parameter param(prep, 10, 10, 1.0f);
    param.W = 0.3f;
    zlsh myZlsh(prep, param, "");

    lsh::timer timer;
    std::cout << std::endl << "RUNNING ZQUERY ..." << std::endl;
    int Qnum = (int)prep.benchmark.N;
    lsh::progress_display pd(Qnum);
    Performance perform;
    for (unsigned j = 0; j < (unsigned)Qnum; j++)
    {
        queryN* q = new queryN(j, c, k, prep, beta);
        //myZlsh.knn(q);
        myZlsh.knnBestFirst(q);
        perform.update(q, prep);
        ++pd;
        delete q; // 修复：避免泄漏
    }

    myZlsh.testLLCP();
    showMemoryInfo();

    float mean_time = (float)perform.timeTotal / perform.num;
    std::cout << "AVG QUERY TIME:    " << mean_time * 1000 << "ms." << std::endl << std::endl;
    std::cout << "AVG RECALL:        " << ((float)perform.NN_num) / (perform.num * k) << std::endl;
    std::cout << "AVG RATIO:         " << ((float)perform.ratio) / (perform.resNum) << std::endl;
    std::cout << "AVG COST:          " << ((float)perform.cost) / ((float)perform.num * prep.data.N) << std::endl;
    std::cout << "\nQUERY FINISH... \n\n\n";

    time_t now = std::time(0);
    time_t zero_point = 1635153971;
    float date = ((float)(now - zero_point)) / 86400;

    std::string fpath = data_fold + "ANN/";

    if (!GenericTool::CheckPathExistence(fpath.c_str())) {
        GenericTool::EnsurePathExistence(fpath.c_str());
        std::cout << BOLDGREEN << "WARNING:\n" << GREEN
                  << "Could not find the path of result file. Have created it. \n"
                  << "The query result will be stored in: " << fpath.c_str() << RESET;
    }
    std::ofstream os(fpath + "ZLSH_result.csv", std::ios_base::app);
    if (os) {
        os.seekp(0, std::ios_base::end);
        int tmp = (int)os.tellp();
        if (tmp == 0) {
            os << "Dataset,c,k,L,K,RATIO,RECALL,AVG_TIME,COST,DATE" << std::endl;
        }
        std::string dataset = datasetName;
        os << dataset << ',' << c << ',' << k << ',' << myLsh.L << ',' << myLsh.K << ','
           << ((float)perform.ratio) / (perform.resNum) << ','
           << ((float)perform.NN_num) / (perform.num * k) << ','
           << mean_time * 1000 << ','
           << ((float)perform.cost) / (perform.num * prep.data.N) << ','
           << date << ','
           << std::endl;
        os.close();
    }
}

bool find_file(std::string&& file)
{
    std::ifstream in(file);
    return in.good();
}

