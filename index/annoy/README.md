1) Annoy 索引的特点与优势（README 核心点）
1. 静态文件索引 + mmap：可跨进程共享

Annoy 最大的特点是能把索引保存成只读文件，并在 load() 时用 mmap 映射到内存，这样多个进程可以共享同一份索引内存页，避免每个进程各存一份。
这对“线上多进程/多 worker 同时查询”非常香。

2. 构建与查询解耦：build 一次，查很多次

Annoy 明确把流程拆成：

先 add_item() 收集数据

再 build() 一次性构建

之后只读查询（不能再 add）

这使得它很适合“离线构建 → 在线服务”的生产模式。

3. 内存占用小、索引文件相对紧凑

README 强调 Annoy 追求小内存/小索引体积，并且把索引做成文件便于分发。

4. 支持多种距离

支持：Euclidean、Manhattan、Cosine（angular）、Hamming、Dot（内积）。
并说明了 angular 与归一化向量欧氏距离的关系。

5. 可用于较高维（但维度越高越难）

README 说在维度不太高时更好（如 <100），但到 1000 维也“出乎意料地工作得不错”。

6. 支持 on-disk build（构建阶段降低内存压力）

on_disk_build() 允许构建时把结构写到文件而不是全放 RAM。

7. 局限性（同样是 README 明说的）

build 后不能再 add（只读）

item id 只接受 整数，并且会按 max(id)+1 分配空间（id 稀疏会浪费内存）

几乎不做 bounds checking，需要自己保证输入合法

2) 参数与 API 的详细说明（逐个讲）

我按“你真正会调的”优先顺序来。

A. 两个最重要的调参：n_trees 与 search_k
1) n_trees（build 时）：精度上限 vs 索引大小/构建时间

在哪里设：

a.build(n_trees, n_jobs=-1)


它影响什么：

build 时间

索引大小

精度（通常树越多越准）

调大：

✅ recall 往上走（更稳）

❌ build 更久、索引更大、内存/磁盘更高

调小：

✅ 索引小、构建快

❌ recall 降、结果更不稳定

README 的建议：

在内存能承受的范围内尽量把 n_trees 设大。

2) search_k（query 时）：查询预算（速度↔精度）

在哪里设：

a.get_nns_by_vector(v, n, search_k=-1, include_distances=False)
a.get_nns_by_item(i, n, search_k=-1, include_distances=False)


它是什么：

查询时“最多检查多少节点/候选”（可以理解为搜索工作量预算）

默认值：
如果不传或 -1：

search_k = n * n_trees（n 是你要的近邻数 top_k）

调大：

✅ recall 提升（更接近真值）

❌ 查询变慢（延迟上升）

调小：

✅ 更快

❌ recall 掉得明显

非常关键的一点（README 明说）：
n_trees 和 search_k 大致独立：
如果你把 search_k 固定住，那么 n_trees 增大不一定导致查询更慢；反之亦然。

README 的建议：

n_trees：内存允许下尽量大

search_k：时间允许下尽量大

B. 必须设置/经常会设置的参数
3) metric（创建索引时）：距离定义（必须对齐你的任务）

在哪里设：

AnnoyIndex(f, metric)


metric 取值："angular", "euclidean", "manhattan", "hamming", "dot"

怎么选：

L2 最近邻：euclidean

余弦相似：angular

README 说明了 angular 与归一化欧氏距离的关系

最大内积：dot

注意：你评测用 Faiss IndexFlatL2 就要配 euclidean；评测 cosine 就要配 angular（并把向量归一化），否则就是“指标错配”。

4) n_jobs（build 时）：建库并行度（性能参数）

在哪里设：

a.build(n_trees, n_jobs=-1)


-1 表示用全部 CPU 核

只影响 build 速度，不影响 recall 的“理论上限”

5) set_seed(seed)（build 前）：可复现实验

在哪里设：

a.set_seed(seed)


只影响建树随机过程

build/load 后再设基本没意义（README 也说只对 build 相关）

C. 索引文件与加载相关参数（生产/大数据常用）
6) save(fn, prefault=False)：保存索引文件

保存后不能再 add

prefault 见下面 load

7) load(fn, prefault=False)：mmap 加载索引文件

关键点：

load 会 mmap，通常很快

prefault 的含义（README 的权衡解释很重要）：

prefault=True：预读整个文件页（冷启动更稳更快）

prefault=False：按需读页（加载快、占用峰值低，但前几次查询可能更慢）

什么时候开：

在线服务、希望冷启动稳定：更倾向 True

内存紧、只查少量、或索引很大：更倾向 False

8) on_disk_build(fn)：在磁盘文件中构建索引

在哪里用：

必须在 add_item 前调用：a.on_disk_build(fn)

用于“构建阶段减少 RAM 压力/支持更大数据”

但它不会让“查询阶段内存”变少——查询时仍然需要 mmap/访问索引页（社区讨论也经常强调这一点）。

D. 查询接口参数（你做实验会经常用到）
9) get_nns_by_vector(v, n, search_k=-1, include_distances=False)

n：你要的 top_k

search_k：上面讲过（查询预算）

include_distances=True：返回 (ids, distances)

10) get_nns_by_item(i, n, search_k=-1, include_distances=False)

同上，只是 query 用已存在的 item id。

E. 其它常用辅助 API（不算“调参”，但很有用）
11) add_item(i, v)

i 必须是非负整数；会按 max(i)+1 分配空间（id 稀疏会浪费）

12) get_item_vector(i) / get_distance(i, j)

调试/验证用。get_distance 现在返回的是“真实距离”（不是旧版本的平方距离）。

13) get_n_items() / get_n_trees() / unload()

检查索引状态、释放 mmap

3) 一套“README 风格”的调参方法（你可直接照做）

固定 metric（和任务一致）

把 n_trees 提到你内存/磁盘能接受的范围（例如 50→100→200）

对每个 n_trees，把 search_k 从默认（-1）开始逐步增大（2x、5x、10x），找到 recall 提升开始变慢的拐点

因为 search_k 就是你愿意花的查询预算

如果要做线上冷启动测试，再决定 prefault 开不开