# Code-to-Code Retrieval Benchmark

This repository contains the full reproducibility package for the paper **"Candidate Recall in Large-Scale Code-to-Code Retrieval: A Comprehensive Benchmark"**, submitted to PVLDB Volume 20.

The benchmark evaluates pre-trained models on the task of **code-to-code retrieval**: given a code snippet as a query, retrieve the most similar snippets from a collection. We evaluate models across four datasets, multiple programming languages, and four code transformation types (LLM, R1, R2, R3).

---

## Requirements

### Hardware

Experiments were conducted on an **NVIDIA DGX Spark AI Workstation** featuring:
- NVIDIA Blackwell GPU (GB10 Grace Blackwell Superchip)
- 128 GB unified LPDDR5x memory
- CUDA 13, PyTorch 2.12.0, cuDNN 9.19

A CUDA-capable GPU is strongly recommended. CPU-only execution is supported but not recommended for large datasets.

### System Dependencies

- Python >= 3.10
- `git` and `git-lfs` (required for xCodeEval)
- Java (required for BigCloneBench H2 database)

### PyTorch

We recommend installing PyTorch separately depending on your system configuration
(CPU/GPU, CUDA version).
Please follow the official instructions at: https://pytorch.org/get-started/locally/

### Python Dependencies
```bash
pip install -r requirements.txt
```

---

## Environment Setup

### 1. Clone the repository
```bash
mkdir benchmark
cd benchmark
git clone https://github.com/leeeov4/code-to-code-benchmark .
```

### 2. Set the base directory

All datasets, processed files, and outputs are stored under a single base directory
controlled by the `BENCHMARK_DATA_DIR` environment variable.
```bash
export BENCHMARK_DATA_DIR=/path/to/your/data/directory
```

Add this line to your `.bashrc` or `.zshrc` to make it permanent.

### 3. Expected directory structure

Once all datasets are downloaded and all experiments are run, the base directory
will have the following structure:
```
$BENCHMARK_DATA_DIR/
├── data/
│   ├── codenet/
|   │   ├── Project_CodeNet_Python800/
|   │   ├── Project_CodeNet_Java250/
|   │   ├── Project_CodeNet_C++1000/
│   ├── multiple/
│   │   ├── parquets/
│   ├── xcodeeval/
│   │   └── retrieval_code_code/
│   └── bigclonebench/
│       ├── h2_db/
│       └── bcb_reduced/
├── processed/
│   ├── codenet/
│   ├── multiple/
│   ├── xcodeeval/
│   └── bigclonebench/
└── output/
    ├── codenet/
    ├── multiple/
    ├── xcodeeval/
    └── bigclonebench/
```

---

## Dataset Download & Preparation

### CodeNet

Download the main archive from IBM:
```bash
wget https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz
```

Then run the preparation script.
```bash
python -m benchmark.scripts.prepare_codenet --archive /path/to/Project_CodeNet.tar.gz
```

### MultiPL-E

Downloads the dataset directly from HuggingFace:
```bash
python -m benchmark.scripts.prepare_multiple
```

### xCodeEval

Downloads the required dataset split from Huggingface:
Requires `git-lfs`.
```bash
python -m benchmark.scripts.prepare_xcodeeval
```

### BigCloneBench

BigCloneBench requires two manual downloads from OneDrive. Please download the
following files and place them in `$BENCHMARK_DATA_DIR/data/bigclonebench/`:

| File | URL |
|------|-----|
| `BigCloneBench_BCEvalVersion.tar.gz` | [Download](https://1drv.ms/u/s!AhXbM6MKt_yLj_NwwVacvUzmi6uorA?e=eMu0P4) |
| `bcb_reduced.tar.gz` | [Download](https://1drv.ms/u/s!AhXbM6MKt_yLj_N15CewgjM7Y8NLKA?e=cScoRJ) |

Then run the preparation script, which extracts the archives and downloads the
H2 database JAR automatically:
```bash
python -m benchmark.scripts.prepare_bigclonebench
```

---

## Running Experiments

Replace `<model_name>` with one of the available model identifiers listed in
[Appendix: Available Models](#appendix-available-models).

Replace `<clone_type>` with one of {type1, type2, type3}.

Replace `<dataset>` with one of {codenet, multiple, xcodeeval}.

### 1. Setup (run once per dataset and language)

Selects and persists the query set for each dataset and language.
For BigCloneBench, extracts and serializes queries and candidates from the SQL database.
```bash
# CodeNet, MultiPL-E, xCodeEval
python -m benchmark.main --dataset <dataset> --stage setup

# BigCloneBench (one per clone type)
python -m benchmark.main --dataset bigclonebench --stage setup [--clone_type <clone_type>]
```

### 2. Full Pipeline (Embeddings + Retrieval + Metrics)

This module executes the complete processing pipeline for each language supported by the selected dataset. The pipeline consists of the following sequential steps:

- Embedding Generation
- Retrieval
- Evaluation Metrics

```bash
# CodeNet, MultiPL-E, xCodeEval
python -m benchmark.main --dataset <dataset> --model <model_name> --stage all
# BigCloneBench
python -m benchmark.main --dataset bigclonebench --model <model_name> --stage all [--clone_type <clone_type>]
```

For a full list of available stages and options, run:
```bash
python -m benchmark.main --help
```

### 3. Embedding Time Analysis
```bash
python -m benchmark.main --analysis embedding_time --model <model_name>
```

### Output Structure

Metrics are saved under `$BENCHMARK_DATA_DIR/output/` with the following structure:
```
$BENCHMARK_DATA_DIR/output/
└── <dataset>/
    └── <language>/
        └── metrics/
```

---

Embedding time results are saved under:
```
$BENCHMARK_DATA_DIR/output/timings/
└── <model_name>_<device>.txt
```

## Code Rewriting

We apply four code rewriting transformations (LLM, R1, R2, R3) to CodeNet and evaluate how model performance changes across all combinations of original and
rewritten queries and candidates. For a detailed description of each transformation, refer to the paper.

### 1. Generate Rewritten Versions (run once per language)

We use Qwen 2.5 Coder model to perform the rewriting.
```bash
python -m benchmark.scripts.transform --language py
python -m benchmark.scripts.transform --language java
python -m benchmark.scripts.transform --language cpp
```

Rewritten versions are saved under `$BENCHMARK_DATA_DIR/processed/codenet/`.

### 2. Embeddings (run once per model and version)
```bash
python -m benchmark.main --dataset codenet --model <model_name> --stage embeddings --version LLM
python -m benchmark.main --dataset codenet --model <model_name> --stage embeddings --version V1
python -m benchmark.main --dataset codenet --model <model_name> --stage embeddings --version V2
python -m benchmark.main --dataset codenet --model <model_name> --stage embeddings --version V3
```

### 3. Retrieval and Metrics (all 25 combinations)
```bash
python -m benchmark.main --dataset codenet --model <model_name> --stage retrieval \
    --query_version all --candidate_version all

python -m benchmark.main --dataset codenet --model <model_name> --stage metrics \
    --query_version all --candidate_version all
```

### Output Structure

Results follow the same structure as the main experiments, with `<query_version>`
and `<candidate_version>` in the filename:
```
$BENCHMARK_DATA_DIR/output/codenet/
└── <language>/
    └── metrics/
        └── <model_name>_<query_version>_<candidate_version>.json
```
 

## Appendix: Available Models

| Model Name | Identifier | HuggingFace |
|---|---|---|
| CodeBERT | `codebert` | [Link](https://huggingface.co/microsoft/codebert-base) |
| StarEncoder | `starencoder` | [Link](https://huggingface.co/bigcode/starencoder) |
| UniXcoder | `unixcoder` | [Link](https://huggingface.co/microsoft/unixcoder-base) |
| CodeT5+ 110M | `codet5p` | [Link](https://huggingface.co/Salesforce/codet5p-110m-embedding) |
| CodeT5+ 220M | `codet5p_220m` | [Link](https://huggingface.co/Salesforce/codet5p-220m) |
| CodexEmbed 400M | `codex` | [Link](https://huggingface.co/Salesforce/SFR-Embedding-Code-400M_R) |
| CodexEmbed 2B | `codex_2b` | [Link](https://huggingface.co/Salesforce/SFR-Embedding-Code-2B_R) |
| ... | ... | ... |


### 1. Generate Rewritten Versions (run once per language)

**Note:** vLLM conflicts with several dependencies used in the main benchmark environment. We recommend running this step in a separate virtual environment.
```bash
# create and activate a dedicated environment
python -m venv venv_rewriting
source venv_rewriting/bin/activate
```

Install PyTorch following the official instructions for your system configuration:
https://pytorch.org/get-started/locally/

Then install the remaining dependencies:
```bash
pip install transformers vllm

# run the transformation script
python -m benchmark.scripts.transform --language py
python -m benchmark.scripts.transform --language java
python -m benchmark.scripts.transform --language cpp

# deactivate when done
deactivate
```

Once the rewritten versions are generated and saved to disk, switch back to the main environment for all subsequent steps (embeddings, retrieval, metrics).