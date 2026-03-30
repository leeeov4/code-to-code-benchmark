# benchmark/scripts/prepare_xcodeeval.py

import os
import shutil
import subprocess
import argparse
from pathlib import Path


REPO_URL   = "https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval"
LFS_INCLUDE = "retrieval_code_code/validation/*"


def run(cmd: list[str], cwd: Path = None):
    print(f"[xCodeEval] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare xCodeEval dataset.")
    parser.add_argument(
        "--keep_clone",
        action="store_true",
        help="Non rimuovere la directory clonata dopo la copia."
    )
    args = parser.parse_args()

    data_dir  = Path(os.environ["BENCHMARK_DATA_DIR"]) / "data"
    clone_dir = data_dir / "xCodeEval_repo"
    out_dir   = data_dir / "xcodeeval" / "retrieval_code_code"

    if out_dir.exists():
        print(f"[xCodeEval] Dataset già presente in {out_dir}, skip.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    # step 1: clone senza LFS
    print("[xCodeEval] Clono repository senza LFS...")
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(
        ["git", "clone", REPO_URL, str(clone_dir)],
        env=env,
        check=True
    )

    # step 2: pull solo della directory necessaria
    print(f"[xCodeEval] Pull LFS: {LFS_INCLUDE}...")
    run(["git", "lfs", "pull", "--include", LFS_INCLUDE], cwd=clone_dir)

    # step 3: copia nella destinazione finale
    src = clone_dir / "retrieval_code_code"
    if not src.exists():
        raise FileNotFoundError(
            f"Directory attesa non trovata: {src}. "
            f"Verifica che il pull LFS sia andato a buon fine."
        )

    print(f"[xCodeEval] Copio in {out_dir}...")
    shutil.copytree(src, out_dir)

    # step 4: rimuovi clone
    if not args.keep_clone:
        print(f"[xCodeEval] Rimuovo directory clone {clone_dir}...")
        shutil.rmtree(clone_dir)

    print(f"[xCodeEval] Dataset pronto in {out_dir}")


if __name__ == "__main__":
    main()