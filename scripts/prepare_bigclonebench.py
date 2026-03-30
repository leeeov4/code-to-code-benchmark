# benchmark/scripts/prepare_bigclonebench.py

import os
import shutil
import tarfile
import urllib.request
import argparse
from pathlib import Path


H2_JAR_URL  = "https://repo1.maven.org/maven2/com/h2database/h2/1.3.176/h2-1.3.176.jar"
H2_JAR_NAME = "h2-1.3.176.jar"

ONEDRIVE_INSTRUCTIONS = """
[BigCloneBench] Manual download required.

Please download the following files and place them in:
$BENCHMARK_DATA_DIR/data/bigclonebench/

1. BigCloneBench_BCEvalVersion.tar.gz:
   URL: https://1drv.ms/u/s!AhXbM6MKt_yLj_NwwVacvUzmi6uorA?e=eMu0P4

2. bcb_reduced.tar.gz:
   URL: https://1drv.ms/u/s!AhXbM6MKt_yLj_N15CewgjM7Y8NLKA?e=cScoRJ

Once done, re-run this script.
"""


def check_archives(bcb_dir: Path) -> tuple[Path, Path]:
    """Cerca i due tar.gz nella directory e li restituisce."""
    bceval_tar  = bcb_dir / "BigCloneBench_BCEvalVersion.tar.gz"
    reduced_tar = bcb_dir / "bcb_reduced.tar.gz"

    missing = []
    if not bceval_tar.exists():
        missing.append("BigCloneBench_BCEvalVersion.tar.gz")
    if not reduced_tar.exists():
        missing.append("bcb_reduced.tar.gz")

    if missing:
        print(f"[BigCloneBench] File mancanti: {missing}")
        print(ONEDRIVE_INSTRUCTIONS)
        raise FileNotFoundError("Archivi mancanti.")

    return bceval_tar, reduced_tar


def extract_bceval(tar_path: Path, h2_db_dir: Path):
    """Estrae BigCloneBench_BCEvalVersion.tar.gz e sposta i 2 file in h2_db/."""
    if h2_db_dir.exists() and any(h2_db_dir.iterdir()):
        print(f"[BigCloneBench] h2_db già presente, skip.")
        return

    h2_db_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tar_path.parent / "tmp_bceval"

    print(f"[BigCloneBench] Estrazione {tar_path.name}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    # i file sono dentro BigCloneBench_BCEvalVersion/
    src_dir = tmp_dir / "BigCloneBench_BCEvalVersion"
    for f in src_dir.iterdir():
        if f.is_file():
            shutil.move(str(f), h2_db_dir / f.name)
            print(f"[BigCloneBench] Spostato {f.name} → {h2_db_dir}")

    shutil.rmtree(tmp_dir)
    tar_path.unlink()


def extract_reduced(tar_path: Path, bcb_dir: Path):
    """Estrae bcb_reduced.tar.gz e sposta bcb_reduced/ in bigclonebench/."""
    out_dir = bcb_dir / "bcb_reduced"
    if out_dir.exists():
        print(f"[BigCloneBench] bcb_reduced già presente, skip.")
        return

    tmp_dir = tar_path.parent / "tmp_reduced"

    print(f"[BigCloneBench] Estrazione {tar_path.name}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    shutil.move(str(tmp_dir / "bcb_reduced"), out_dir)
    print(f"[BigCloneBench] Spostato bcb_reduced → {out_dir}")

    shutil.rmtree(tmp_dir)
    tar_path.unlink()


def download_h2_jar(h2_db_dir: Path):
    """Scarica il JAR H2 da Maven Central."""
    jar_path = h2_db_dir / H2_JAR_NAME
    if jar_path.exists():
        print(f"[BigCloneBench] {H2_JAR_NAME} già presente, skip.")
        return
    print(f"[BigCloneBench] Scarico {H2_JAR_NAME} da Maven Central...")
    urllib.request.urlretrieve(H2_JAR_URL, jar_path)
    print(f"[BigCloneBench] Salvato in {jar_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare BigCloneBench dataset.")
    parser.parse_args()

    bcb_dir   = Path(os.environ["BENCHMARK_DATA_DIR"]) / "data" / "bigclonebench"
    h2_db_dir = bcb_dir / "h2_db"
    bcb_dir.mkdir(parents=True, exist_ok=True)

    bceval_tar, reduced_tar = check_archives(bcb_dir)

    extract_bceval(bceval_tar, h2_db_dir)
    extract_reduced(reduced_tar, bcb_dir)
    download_h2_jar(h2_db_dir)

    print(f"\n[BigCloneBench] Setup completato:")
    print(f"  h2_db:       {h2_db_dir}")
    print(f"  bcb_reduced: {bcb_dir / 'bcb_reduced'}")


if __name__ == "__main__":
    main()