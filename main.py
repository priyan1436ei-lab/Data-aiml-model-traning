"""
Universal Global Coding Dataset - Main Platform CLI
Coordinates discovery, collection, cleaning, deduplication, tokenization, training preparation, and reporting.
"""
import argparse
import sys
from pathlib import Path

# Add workspace root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import init_workspace, PATHS
from src.discovery.manager import DiscoveryManager
from src.acquisition.manager import AcquisitionManager
from src.quality.manager import QualityManager
from src.deduplication.manager import DeduplicationManager
from src.tokenization.manager import TokenizationManager
from src.normalization.normalizer import TrainingCorpusNormalizer
from src.export.exporter import DatasetExporter
from src.evaluation.report_generator import ReportGenerator

def cmd_discover(args):
    print("=====================================================================")
    print("STAGE 1: GLOBAL DATASET DISCOVERY")
    print("=====================================================================")
    manager = DiscoveryManager()
    sources = manager.discover_all(enable_live_apis=not args.offline, query_limit=args.limit)
    print(f"Discovery complete. Total sources registered: {len(sources)}")

def cmd_collect(args):
    print("=====================================================================")
    print("STAGE 2: DATASET ACQUISITION & INGESTION")
    print("=====================================================================")
    manager = AcquisitionManager()
    records = manager.run_acquisition(fetch_remote_benchmarks=not args.offline)
    print(f"Acquisition complete. Total raw records acquired: {len(records)}")

def cmd_clean(args):
    print("=====================================================================")
    print("STAGE 3: DATA QUALITY ASSESSMENT & CLEANING")
    print("=====================================================================")
    manager = QualityManager()
    summary = manager.run_cleaning_and_quality_assessment()
    print("Quality assessment and cleaning complete.")

def cmd_deduplicate(args):
    print("=====================================================================")
    print("STAGE 4: MULTI-STAGE DEDUPLICATION (EXACT, NORMALIZED, MINHASH LSH)")
    print("=====================================================================")
    manager = DeduplicationManager()
    summary = manager.run_deduplication()
    print("Deduplication complete.")

def cmd_tokenize(args):
    print("=====================================================================")
    print("STAGE 5: TOKENIZATION, SEQUENCE PACKING & SPLITTING")
    print("=====================================================================")
    manager = TokenizationManager(tokenizer_model=args.tokenizer)
    report = manager.run_tokenization()
    print("Tokenization and sequence packing complete.")

def cmd_prepare_training(args):
    print("=====================================================================")
    print("STAGE 6: TRAINING CORPUS NORMALIZATION & PARQUET EXPORT")
    print("=====================================================================")
    # 1. Normalize into specific training sub-corpora
    normalizer = TrainingCorpusNormalizer()
    normalizer.run_normalization()

    # 2. Export all JSONL datasets to Parquet
    print("[Export] Exporting all training corpora into compressed Parquet format...")
    DatasetExporter.export_all_corpora()
    print("Training preparation complete.")

def cmd_evaluate(args):
    print("=====================================================================")
    print("STAGE 7: QUALITY & COMPLIANCE EVALUATION")
    print("=====================================================================")
    import json
    q_summary_path = PATHS["reports"] / "quality_assessment_summary.json"
    d_summary_path = PATHS["reports"] / "deduplication_summary.json"
    t_summary_path = PATHS["reports"] / "tokenization_report.json"
    inv_path = PATHS["metadata"] / "discovered_datasets.json"

    q_stats = json.load(open(q_summary_path, "r", encoding="utf-8")) if q_summary_path.exists() else {}
    d_stats = json.load(open(d_summary_path, "r", encoding="utf-8")) if d_summary_path.exists() else {}
    t_stats = json.load(open(t_summary_path, "r", encoding="utf-8")) if t_summary_path.exists() else {}
    sources = json.load(open(inv_path, "r", encoding="utf-8")) if inv_path.exists() else []

    print(f"Total Sources Discovered: {len(sources)}")
    print(f"Total Raw Records Evaluated: {q_stats.get('total_processed', 0)}")
    print(f"Clean Records: {q_stats.get('total_cleaned', 0)} ({q_stats.get('pass_rate_percentage', 0)}%)")
    print(f"Duplicates Removed: {d_stats.get('total_duplicates_removed', 0)} ({d_stats.get('duplicate_reduction_percentage', 0)}%)")
    print(f"Total Measured Tokens: {t_stats.get('total_tokens', 0):,}")
    print("Evaluation summary printed.")

def cmd_report(args):
    print("=====================================================================")
    print("STAGE 8: REPORTING & DOCUMENTATION GENERATION")
    print("=====================================================================")
    import json
    q_summary_path = PATHS["reports"] / "quality_assessment_summary.json"
    d_summary_path = PATHS["reports"] / "deduplication_summary.json"
    t_summary_path = PATHS["reports"] / "tokenization_report.json"
    inv_path = PATHS["metadata"] / "discovered_datasets.json"

    q_stats = json.load(open(q_summary_path, "r", encoding="utf-8")) if q_summary_path.exists() else {}
    d_stats = json.load(open(d_summary_path, "r", encoding="utf-8")) if d_summary_path.exists() else {}
    t_stats = json.load(open(t_summary_path, "r", encoding="utf-8")) if t_summary_path.exists() else {}
    sources = json.load(open(inv_path, "r", encoding="utf-8")) if inv_path.exists() else []

    ReportGenerator.generate_all_reports(
        token_stats=t_stats,
        quality_stats=q_stats,
        dedup_stats=d_stats,
        discovery_count=len(sources)
    )
    print("All reports and manifests generated successfully.")

def cmd_run_all(args):
    """Run full end-to-end dataset preparation pipeline and model training."""
    print("#####################################################################")
    print("STARTING UNIVERSAL CODING PLATFORM: FULL END-TO-END EXECUTION")
    print("#####################################################################\n")
    cmd_discover(args)
    cmd_collect(args)
    cmd_clean(args)
    cmd_deduplicate(args)
    cmd_tokenize(args)
    cmd_prepare_training(args)
    cmd_evaluate(args)
    cmd_report(args)

    if not getattr(args, "skip_train", False):
        print("\n=====================================================================")
        print("STAGE 9: TRANSFORMER MODEL TRAINING")
        print("=====================================================================")
        cmd_train(args)

    print("\n#####################################################################")
    print("[SUCCESS] Full Universal Coding Pipeline and Model Training Completed!")
    print("#####################################################################")

def cmd_pipeline(args):
    """Run full end-to-end dataset pipeline in sequence."""
    cmd_run_all(args)

def cmd_serve(args):
    """Launch the FastAPI REST API and Explorer service."""
    import uvicorn
    print(f"Starting Universal Dataset API server on http://{args.host}:{args.port}")
    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=False)

def cmd_train(args):
    """Launch Transformer Coding Model training on the prepared dataset."""
    from train import train_model
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        seq_len=args.seq_len,
        device_str=args.device
    )

def main():
    parser = argparse.ArgumentParser(
        description="Universal Global Coding Dataset Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # discover
    p_disc = subparsers.add_parser("discover", help="Discover coding datasets across HF, GitHub, and Zenodo")
    p_disc.add_argument("--limit", type=int, default=15, help="Max items per ecosystem")
    p_disc.add_argument("--offline", action="store_true", help="Run catalog discovery only without network APIs")

    # collect
    p_coll = subparsers.add_parser("collect", help="Acquire datasets with SHA-256 provenance tracking")
    p_coll.add_argument("--offline", action="store_true", help="Run local seed collection only without network APIs")

    # clean
    subparsers.add_parser("clean", help="Assess quality, syntax validity, PII/secrets, and benchmark contamination")

    # deduplicate
    subparsers.add_parser("deduplicate", help="Deduplicate records (Exact SHA-256, Normalized, MinHash LSH)")

    # tokenize
    p_tok = subparsers.add_parser("tokenize", help="Count tokens, analyze sequence lengths, pack contexts, and split")
    p_tok.add_argument("--tokenizer", type=str, default="fast_bpe", help="Tokenizer model name or 'fast_bpe'")

    # prepare-training
    subparsers.add_parser("prepare-training", help="Normalize schemas and export into JSONL and compressed Parquet")

    # evaluate
    subparsers.add_parser("evaluate", help="Print quality, deduplication, and training metrics")

    # report
    subparsers.add_parser("report", help="Generate DATASET_CARD.md, QUALITY_REPORT.json, and TRAINING_MANIFEST.json")

    # train
    p_trn = subparsers.add_parser("train", help="Train a causal transformer coding model on the dataset")
    p_trn.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    p_trn.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p_trn.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    p_trn.add_argument("--max-steps", type=int, default=30, help="Maximum gradient update steps")
    p_trn.add_argument("--seq-len", type=int, default=128, help="Context sequence length")
    p_trn.add_argument("--device", type=str, default=None, help="Device: 'cpu' or 'cuda'")

    # run-all / pipeline (all-in-one)
    for alias_name in ["run-all", "pipeline"]:
        p_all = subparsers.add_parser(alias_name, help="Execute entire end-to-end dataset pipeline and model training")
        p_all.add_argument("--limit", type=int, default=15, help="Max items per ecosystem")
        p_all.add_argument("--offline", action="store_true", help="Run without network calls")
        p_all.add_argument("--tokenizer", type=str, default="fast_bpe", help="Tokenizer model")
        p_all.add_argument("--skip-train", action="store_true", help="Skip model training stage")
        p_all.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
        p_all.add_argument("--batch-size", type=int, default=4, help="Batch size")
        p_all.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
        p_all.add_argument("--max-steps", type=int, default=30, help="Maximum gradient update steps")
        p_all.add_argument("--seq-len", type=int, default=128, help="Context sequence length")
        p_all.add_argument("--device", type=str, default=None, help="Device: 'cpu' or 'cuda'")

    # serve
    p_srv = subparsers.add_parser("serve", help="Launch FastAPI web server")
    p_srv.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    p_srv.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    init_workspace()

    command_map = {
        "discover": cmd_discover,
        "collect": cmd_collect,
        "clean": cmd_clean,
        "deduplicate": cmd_deduplicate,
        "tokenize": cmd_tokenize,
        "prepare-training": cmd_prepare_training,
        "evaluate": cmd_evaluate,
        "report": cmd_report,
        "train": cmd_train,
        "run-all": cmd_run_all,
        "pipeline": cmd_run_all,
        "serve": cmd_serve
    }

    cmd_fn = command_map.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

