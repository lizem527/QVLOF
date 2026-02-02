# 🧠 QVLOF: A Query-driven Vector Layout Optimization Framework for I/O Efficient Approximate Nearest Neighbor Search

This repository contains the source code for **QVLOF**, a **Query-driven Vector Layout Optimization Framework** designed to improve **I/O efficiency** in **k-Approximate Nearest Neighbor (k-ANN) search** under both **static** and **dynamic** workloads.

🔗 **Project Page**: https://github.com/lizem527/QVLOF

---

## 📖 Project Overview

In the k-ANN search workflow, modern vector indexes can achieve high search performance under static or dynamic workloads. However, search results are often scattered across multiple physical blocks, leading to excessive I/O cost.

Layout optimization provides a promising approach to reduce I/O cost by reordering vectors in physical storage. Unfortunately, due to the sparse distribution of vectors in high-dimensional space and the limited information available from k-ANN queries, existing methods struggle to colocate vectors that are both **similar** and **frequently accessed**.

To address this challenge, we propose **QVLOF**, a **query-driven vector layout optimization framework** that supports I/O-efficient k-ANN search under both static and dynamic workloads:

- Under **static workloads**, QVLOF introduces **Query-attracted Similarity-based Ordering (QSO)**.
- Under **dynamic workloads**, QVLOF further designs **Representative-query Distance-based Ordering (RDO)**.
- The **decoupled design** of QVLOF enables seamless integration with existing vector indexes.

------

## 📂 Repository Structure

```text
QVLOF
├── data                     # Dataset download and preprocessing
│   └── dataset_download.py
├── index                    # Vector index implementations
│   ├── annoy
│   ├── det-lsh
│   ├── hnsw
│   ├── ivfpq
│   ├── lsh-apg
│   ├── mirage
│   ├── ogp
│   └── pg_vector
├── qso                      # Query-attracted Similarity-based Ordering (QSO)
│   ├── demo.py              # One-click layout generation
│   └── adacurve.py          # Training-related code for AdaCurve
├── rdo                      # Representative-query Distance-based Ordering (RDO)
├── requirements             # Conda / pip environment configurations
├── results                  # Generated layouts and query CSV files
│   ├── fn···.csv            # Reordered data CSV files
│   └── select_query.csv     # Query vector CSV file
└── scripts                  # One-click experiment scripts
    ├── dynamic_memory_sh    # Dynamic in-memory experiments
    ├── static_memory_sh     # Static in-memory experiments
    └── clac_block_py        # Cross-block access analysis
```

------

## 🌲 Environment Setup (qvlof_env)

> **Tested on Ubuntu 22.04**

### GCC (>=13)

```sh
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-13 g++-13

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 90
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

### CMake (Latest)

```sh
sudo apt remove -y cmake
sudo apt install -y ca-certificates gnupg software-properties-common

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
 | sudo gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] \
https://apt.kitware.com/ubuntu/ jammy main" \
| sudo tee /etc/apt/sources.list.d/kitware.list

sudo apt update
sudo apt install -y cmake
cmake --version
sudo ln -s /usr/bin/cmake /usr/local/bin/cmake
```

### Clang & OpenMP

```sh
sudo apt install -y clang-14 libomp5 libomp-dev
clang-14 -fopenmp -v -x c - <<<'int main(){return 0;}'
```

### Math & System Dependencies

```sh
sudo apt install -y \
  build-essential \
  libopenblas-dev \
  liblapack-dev \
  pkg-config \
  ninja-build \
  libboost-dev \
  libboost-math-dev \
  libsparsehash-dev \
  swig python3-dev python3-numpy \
  git numactl
```

### Python Environment (qvlof_env)

```sh
conda env create -f vector_db.env.yml
conda activate qvlof_env
```

Install Python dependencies:

```sh
# Base tools
python -m pip install -U pip setuptools wheel

# In-memory experiments
pip install faiss-cpu==1.11.0 numpy==1.26.4 annoy

# Disk-based experiments
python -m pip install -r pg_vector.requirements.txt

# QSO (learning-based components)
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

python -m pip install -r dnn.requirements.cpu.txt
```

------

## 📊 Datasets

| Dataset     | Scale | Dimension |
| ----------- | ----- | --------- |
| **DEEP-1M** | 1M    | 96        |
| **SIFT-1M** | 1M    | 128       |
| **GIST**    | 1M    | 960       |

### Download

```bash
cd data
python dataset_download.py
```

------

## 🧩 Static Layout Optimization (QSO)

### Query-attracted Similarity-based Ordering

The **QSO** module implements static, query-driven vector layout optimization.

### Supported Methods

- **QSO (ours)**
- **Baselines**:
  - `z-order`
  - `hilbert`
  - `idistance`
  - `adacurve`
  - `random`

The script `qso/demo.py` provides a **one-click interface** to generate reordered layouts for all the above methods.

```bash
python qso/demo.py
```

### Note on AdaCurve

- **AdaCurve** is a learning-based layout optimization method proposed in prior work and requires training.
- This repository provides **training-related code only** in:

```text
python qso/adacurve.py
```

------

## 🔄 Dynamic Layout Optimization (RDO)

### Representative-query Distance-based Ordering

> **All `[config_name]` correspond to configuration files under the `params/` directory.**

The RDO pipeline consists of three stages:

1. **Generate partial candidate data layouts**

```sh
python RDO_layout_main.py --config config_name
```

2. **RDO switch strategy and I/O cost**

```sh
python RDO_switch_main.py --config config_name
```

3. **Measure end-to-end runtime**

```sh
python RDO_replay_main.py --config config_name --rewrite --root /path/to/partition --alg random
```

## 🧪 In-Memory Experiments

QVLOF supports both **static** and **dynamic** in-memory experiments on the following vector indexes:

**MIRAGE, HNSW, IVF-PQ, LSH-APG, ANNOY, DET-LSH, OGP**

Each index provides a one-click execution script under:

```text
scripts/static_memory_sh/
scripts/dynamic_memory_sh/
```

### Example (HNSW, Dynamic)

```bash
cd index/hnsw/build
bash dynamic_query_hnsw_benchmark.sh
```

------

## 💾 Disk-Based Experiments (pgvector)

QVLOF supports disk-based ANN search using **pgvector**.

### Index Construction

```bash
cd index/pg_vector
conda activate qvlof_env
sudo /etc/init.d/postgresql restart
python load_and_index.py
```

### Query Evaluation

```bash
sudo /etc/init.d/postgresql restart
python run_experiment_ubuntu_final.py
```

Metrics include **query latency**, **recall**, and **QPS**.

------

## 📐 Cross-Block Access Analysis

Scripts for analyzing **cross-block access rate** are provided in:

```text
scripts/clac_block_py/
```

### Example

```bash
conda activate qvlof_env
python block_ivfpq_static.py
python block_tools_dynamic.py
```

------

## 📧 Contact

For questions or issues, please contact:

📩 **[lizeming@bit.edu.cn](mailto:lizeming@bit.edu.cn)**

