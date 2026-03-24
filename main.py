# benchmark/main.py

import argparse
import itertools

from .config import BCB_DB_PATH, SEED, OUTPUT_PATH
from .core.base_dataset import BaseDataset
from .datasets.multiple import MultiPLE
from .datasets.codenet import CodeNet
from .datasets.xcodeeval import xCodeEval
from .datasets.bigclonebench import BigCloneBench
from .pipeline.pipeline import Pipeline
from .models.starencoder import StarEncoder
from .models.codebert import CodeBERT
from .models.codet5 import CodeT5


# ------------------------------------------------------------------ #
#  Registry                                                            #
# ------------------------------------------------------------------ #

def build_dataset(args) -> BaseDataset:
    if args.dataset == "codenet":       return CodeNet()
    if args.dataset == "multiple":     return MultiPLE()
    if args.dataset == "xcodeeval":     return xCodeEval()
    if args.dataset == "bigclonebench": return BigCloneBench(clone_type=args.clone_type)
    raise ValueError(f"Dataset sconosciuto: {args.dataset}")



def build_model(args):
    if args.model == "starencoder":
        return StarEncoder()
    elif args.model == "codebert":
        return CodeBERT()
    elif args.model == "codet5":
        return CodeT5()
    elif args.model == "codet5_large":
        return CodeT5("large")

    # aggiungere qui i modelli man mano
    raise NotImplementedError(f"Modello non registrato: {args.model}")


# ------------------------------------------------------------------ #
#  Stage handlers                                                      #
# ------------------------------------------------------------------ #

def run_setup(dataset: BaseDataset, args):
    if isinstance(dataset, BigCloneBench):
        dataset.extract_and_serialize()
        print("[Setup] Estrazione BigCloneBench completata.")
    else:
        for lang in _resolve_languages(dataset, args):
            dataset.select_queries(lang, seed=SEED)
            print(f"[Setup] Query selezionate per {lang}.")


def run_transform(dataset: BaseDataset, args):
    from .transformers.transformer_factory import build_transformer
    if args.version is None:
        raise ValueError("--version è obbligatorio per lo stage transform.")
    transformer = build_transformer(args.version)
    for lang in _resolve_languages(dataset, args):
        dataset.transform_and_serialize(lang, args.version, transformer)
        print(f"[Transform] Versione {args.version} generata per {lang}.")


def run_embeddings(dataset: BaseDataset, pipeline: Pipeline, args):
    version = args.version or "original"
    for lang in _resolve_languages(dataset, args):
        pipeline.run_stage1_embeddings(lang, version)


def run_retrieval(dataset: BaseDataset, pipeline: Pipeline, args):
    q_versions = _resolve_versions(args.query_version)
    c_versions = _resolve_versions(args.candidate_version)
    for lang in _resolve_languages(dataset, args):
        for q_ver, c_ver in itertools.product(q_versions, c_versions):
            print(f"[Retrieval] {lang} | query={q_ver} candidates={c_ver}")
            pipeline.run_stage2_retrieval(lang, q_ver, c_ver)


def run_metrics(dataset: BaseDataset, pipeline: Pipeline, args):
    q_versions = _resolve_versions(args.query_version)
    c_versions = _resolve_versions(args.candidate_version)
    for lang in _resolve_languages(dataset, args):
        for q_ver, c_ver in itertools.product(q_versions, c_versions):
            print(f"[Metrics] {lang} | query={q_ver} candidates={c_ver}")
            summary = pipeline.run_stage3_metrics(lang, q_ver, c_ver)
            _print_metrics(summary)


# ------------------------------------------------------------------ #
#  Utility                                                             #
# ------------------------------------------------------------------ #

VERSIONS = ["original", "LLM", "R1", "R2", "R3"]


def _resolve_languages(dataset: BaseDataset, args) -> list[str]:
    """Restituisce il singolo linguaggio richiesto o tutti quelli supportati."""
    if args.language:
        if args.language not in dataset.supported_languages():
            raise ValueError(
                f"{args.language} non supportato da {dataset.__class__.__name__}. "
                f"Supportati: {dataset.supported_languages()}"
            )
        return [args.language]
    return dataset.supported_languages()


def _resolve_versions(version_arg: str) -> list[str]:
    """'all' espande a tutte le versioni, altrimenti restituisce la versione richiesta."""
    if version_arg == "all":
        return VERSIONS
    if version_arg not in VERSIONS:
        raise ValueError(f"Versione sconosciuta: {version_arg}. Disponibili: {VERSIONS}")
    return [version_arg]


def _print_metrics(summary: dict):
    for metric, value in summary.items():
        print(f"  {metric}: {value:.4f}")


# ------------------------------------------------------------------ #
#  Argparse                                                            #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description="Code-to-code retrieval benchmark.")

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["codenet", "multiple", "xcodeeval", "bigclonebench"]
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["setup", "transform", "embeddings", "retrieval", "metrics", "all"]
    )
    parser.add_argument(
        "--model",
        required=False  # non obbligatorio per setup e transform
    )
    parser.add_argument(
        "--language",
        required=False,
        help="Se omesso, esegue su tutti i linguaggi supportati dal dataset."
    )
    parser.add_argument(
        "--version",
        required=False,
        help="Versione del codice (original, LLM, R1, R2, R3). Usato per embeddings e transform."
    )
    parser.add_argument(
        "--query_version",
        default="original",
        choices=VERSIONS + ["all"],
        help="Versione delle query per retrieval e metrics."
    )
    parser.add_argument(
        "--candidate_version",
        default="original",
        choices=VERSIONS + ["all"],
        help="Versione dei candidati per retrieval e metrics."
    )
    parser.add_argument(
        "--clone_type",
        required=False,
        choices=["type1", "type2", "type3"],
        help="Obbligatorio per bigclonebench."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=False,
        default=None,
        help="Se specificato, salva solo i top-k candidati per ogni query."
    )

    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    dataset = build_dataset(args)

    # setup e transform non richiedono un modello
    if args.stage == "setup":
        run_setup(dataset, args)
        return

    if args.stage == "transform":
        run_transform(dataset, args)
        return

    # tutti gli altri stage richiedono un modello
    if args.model is None:
        raise ValueError("--model è obbligatorio per gli stage embeddings, retrieval, metrics.")

    model = build_model(args)
    pipeline = Pipeline(dataset, model, OUTPUT_PATH, top_k=args.top_k)

    if args.stage == "embeddings":
        run_embeddings(dataset, pipeline, args)

    elif args.stage == "retrieval":
        run_retrieval(dataset, pipeline, args)

    elif args.stage == "metrics":
        run_metrics(dataset, pipeline, args)

    elif args.stage == "all":
        version = args.version or "original"
        for lang in _resolve_languages(dataset, args):
            pipeline.run_stage1_embeddings(lang, version)
        run_retrieval(dataset, pipeline, args)
        run_metrics(dataset, pipeline, args)


if __name__ == "__main__":
    main()