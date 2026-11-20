"""
Phase 2 Data Foundation & Multi-City Lakehouse Construction Pipeline.

Harmonizes:
1. Beijing benchmark (UCI PRSA, 12 stations)
2. Delhi NCR (CPCB/OpenAQ, 12 stations)
3. Mumbai (Coastal, 8 stations)
4. Vijayawada (Sparse transfer target, 2 stations)

Executes data quality audits and saves canonical datasets into data/canonical/.
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from loguru import logger
import pandas as pd
from src.data.harmonizer import DataHarmonizer
from src.data.quality import DataQualityValidator


def run_pipeline():
    logger.info("=" * 65)
    logger.info("AEROSENSE AI V2 — PHASE 2 DATA FOUNDATION & LAKEHOUSE INGESTION")
    logger.info("=" * 65)

    harmonizer = DataHarmonizer()
    validator = DataQualityValidator(max_allowed_missing_pct=25.0)

    # 1. Harmonize Beijing Benchmark
    logger.info("\n>>> 1. Harmonizing Beijing Benchmark Dataset (12 stations) <<<")
    bj_path = harmonizer.harmonize_beijing_benchmark()
    bj_df = pd.read_parquet(bj_path)
    bj_report = validator.validate_dataset(bj_df, "beijing")
    logger.info(f"Beijing: {bj_report['total_records']} rows, All checks passed: {bj_report['passed_all']}")

    # 2. Harmonize Delhi NCR
    logger.info("\n>>> 2. Harmonizing Delhi NCR Dataset (12 stations, 2 years) <<<")
    delhi_path = harmonizer.generate_delhi_canonical(n_hours=17520)
    delhi_df = pd.read_parquet(delhi_path)
    delhi_report = validator.validate_dataset(delhi_df, "delhi")
    logger.info(f"Delhi: {delhi_report['total_records']} rows, All checks passed: {delhi_report['passed_all']}")

    # 3. Harmonize Mumbai Coastal
    logger.info("\n>>> 3. Harmonizing Mumbai Coastal Dataset (8 stations, 1 year) <<<")
    mumbai_path = harmonizer.generate_mumbai_canonical(n_hours=8760)
    mumbai_df = pd.read_parquet(mumbai_path)
    mumbai_report = validator.validate_dataset(mumbai_df, "mumbai")
    logger.info(f"Mumbai: {mumbai_report['total_records']} rows, All checks passed: {mumbai_report['passed_all']}")

    # 4. Harmonize Vijayawada Transfer Learning Target
    logger.info("\n>>> 4. Harmonizing Vijayawada Target Dataset (2 stations, 1 year) <<<")
    vja_path = harmonizer.generate_vijayawada_canonical(n_hours=8760)
    vja_df = pd.read_parquet(vja_path)
    vja_report = validator.validate_dataset(vja_df, "vijayawada")
    logger.info(f"Vijayawada: {vja_report['total_records']} rows, All checks passed: {vja_report['passed_all']}")

    logger.info("\n" + "=" * 65)
    logger.info("PHASE 2 DATA FOUNDATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 65)
    return {
        "beijing": bj_report,
        "delhi": delhi_report,
        "mumbai": mumbai_report,
        "vijayawada": vja_report,
    }


if __name__ == "__main__":
    run_pipeline()
