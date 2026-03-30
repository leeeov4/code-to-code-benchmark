# benchmark/scripts/prepare_codenet.py

import os
import random
import shutil
import tarfile
import argparse
from pathlib import Path
from tqdm import tqdm
from ..config import SEED

DATASETS = {
    "python":   "Project_CodeNet_Python800",
    "java": "Project_CodeNet_Java250",
    "cpp":  "Project_CodeNet_C++1000",
}

MAIN_ARCHIVE    = "Project_CodeNet.tar.gz"
BENCHMARKS_PATH = "Project_CodeNet/derived/benchmarks"

N_PROBLEMS    = 250
N_SUBMISSIONS = 300


def extract_main_archive(archive_path: Path, out_dir: Path):
    """Estrae solo gli archivi dataset da derived/benchmarks nella destinazione finale."""
    print(f"[CodeNet] Estrazione archivio principale: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [
            m for m in tar.getmembers()
            if m.name.startswith(BENCHMARKS_PATH) and m.name.endswith(".tar.gz")
        ]
        print(f"[CodeNet] Trovati {len(members)} archivi dataset.")
        tar.extractall(path=out_dir, members=members)


def extract_dataset_archive(archive_path: Path, out_dir: Path):
    """Estrae un singolo archivio dataset direttamente in out_dir."""
    print(f"[CodeNet] Estrazione {archive_path.name}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=out_dir)
    archive_path.unlink()  # rimuovi il tar.gz dopo l'estrazione


def trim_dataset(dataset_dir: Path, seed: int):
    """Rimuove problemi e submission in eccesso direttamente in dataset_dir."""
    rng      = random.Random(seed)
    problems = [p for p in dataset_dir.iterdir() if p.is_dir()]

    # rimuovi problemi in eccesso
    if len(problems) > N_PROBLEMS:
        to_keep   = set(p.name for p in rng.sample(problems, N_PROBLEMS))
        to_remove = [p for p in problems if p.name not in to_keep]
        print(f"[CodeNet] Rimuovo {len(to_remove)} problemi in eccesso da {dataset_dir.name}...")
        for problem_dir in tqdm(to_remove, desc="Removing problems"):
            shutil.rmtree(problem_dir)
        problems = [p for p in dataset_dir.iterdir() if p.is_dir()]

    # rimuovi submission in eccesso per ogni problema
    for problem_dir in tqdm(problems, desc=f"Trimming {dataset_dir.name}"):
        submissions = [s for s in problem_dir.iterdir() if s.is_file()]
        if len(submissions) > N_SUBMISSIONS:
            to_keep   = set(s.name for s in rng.sample(submissions, N_SUBMISSIONS))
            to_remove = [s for s in submissions if s.name not in to_keep]
            for submission in to_remove:
                submission.unlink()


def main():
    parser = argparse.ArgumentParser(description="Prepare CodeNet dataset.")
    parser.add_argument(
        "--archive",
        required=True,
        help="Path to Project_CodeNet.tar.gz"
    )
    args = parser.parse_args()

    #data_dir = Path(os.environ["BENCHMARK_DATA_DIR"]) / "data"
    data_dir = Path(os.environ["BENCHMARK_DATA_DIR"]) / "data" / "codenet"
    data_dir.mkdir(parents=True, exist_ok=True)

    # step 1: estrai archivi dataset da archivio principale
    extract_main_archive(Path(args.archive), data_dir)

    # step 2: estrai e trimma ogni dataset
    benchmarks_dir = data_dir / BENCHMARKS_PATH
    for lang, dataset_name in DATASETS.items():
        archive_path = benchmarks_dir / f"{dataset_name}.tar.gz"
        if not archive_path.exists():
            print(f"[CodeNet] WARN: {archive_path} non trovato, skip.")
            continue

        dataset_dir = data_dir / dataset_name
        if dataset_dir.exists():
            print(f"[CodeNet] {dataset_name} già presente, skip.")
            continue

        extract_dataset_archive(archive_path, data_dir)
        trim_dataset(dataset_dir, seed=SEED)

    # rimuovi directory benchmarks ormai vuota
    shutil.rmtree(data_dir / "Project_CodeNet", ignore_errors=True)

    print(f"[CodeNet] Dataset pronti in {data_dir}")


if __name__ == "__main__":
    main()

"""
Il flusso ora è:
Project_CodeNet.tar.gz
    → estrai *.tar.gz direttamente in $BENCHMARK_DATA_DIR/data/
    → per ogni dataset: estrai in-place, rimuovi problemi in eccesso, rimuovi submission in eccesso
    → rimuovi directory Project_CodeNet/ residua
"""