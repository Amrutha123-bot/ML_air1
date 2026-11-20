import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AeroSense AI V2 — 01: Dataset Acquisition & Quality Inspection\n",
                "\n",
                "> **Execution Status:** `USER MUST EXECUTE IN COLAB`  \n",
                "> **Phase:** Phase 4 (Dataset Acquisition and Validation)  \n",
                "> **Outputs:** `reports/dataset_inventory.csv`, `docs/DATASET_VALIDATION_REPORT.md`  \n",
                "\n",
                "This notebook loads raw and canonical observations across Indian metropolitan monitoring networks (CPCB, OpenCity, OpenAQ), executes forensic quality checks (duplicate detection, missing value percentages, gap histograms, 144h usable windows), and enforces the 14-point scientific integrity audit.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Environment & Dependency Setup\n",
                "!pip install -q pandas numpy pyarrow fastparquet loguru scikit-learn matplotlib seaborn\n",
                "\n",
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "\n",
                "try:\n",
                "    from google.colab import drive\n",
                "    drive.mount('/content/drive')\n",
                "    ARTIFACT_ROOT = Path('/content/drive/MyDrive/AeroSense_AI_Artifacts')\n",
                "    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)\n",
                "    print(f'Artifact root mounted at: {ARTIFACT_ROOT}')\n",
                "except Exception:\n",
                "    ARTIFACT_ROOT = Path('./artifacts')\n",
                "    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)\n",
                "    print('Running in local environment')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Ingestion & Quality Inspection across Indian Cities\n",
                "from aerosense_ml.data.loaders import load_canonical_lakehouse\n",
                "from aerosense_ml.data.inspection import inspect_city_data, generate_dataset_inventory\n",
                "from aerosense_ml.evaluation.integrity import PredictionIntegrityAuditor\n",
                "\n",
                "CITIES = [\n",
                "    ('delhi', 'CPCB CAAQMS 12-Station Network'),\n",
                "    ('mumbai', 'CPCB / MPCB Air Quality Monitoring Network'),\n",
                "    ('vijayawada', 'APPCB Ambient Air Monitoring Programme'),\n",
                "]\n",
                "\n",
                "inspections = []\n",
                "for city, src in CITIES:\n",
                "    try:\n",
                "        df = load_canonical_lakehouse(city)\n",
                "        insp = inspect_city_data(df, city_name=city, source_name=src)\n",
                "        inspections.append(insp)\n",
                "        print(f\"[OK] {city.upper()}: {insp['total_rows']:,} rows, {insp['station_count']} stations -> Status: {insp['status']}\")\n",
                "    except Exception as e:\n",
                "        print(f\"[ERR] Could not inspect {city}: {e}\")\n",
                "\n",
                "inv_df = generate_dataset_inventory(inspections, output_path='reports/dataset_inventory.csv')\n",
                "print('\\n--- Generated Dataset Inventory ---')\n",
                "print(inv_df.to_string())\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. 14-Point Scientific Integrity Audit\n",
                "auditor = PredictionIntegrityAuditor()\n",
                "audit_results = auditor.audit_all()\n",
                "for point, res in audit_results.items():\n",
                "    status_icon = 'PASS' if res['status'] == 'PASS' else 'FAIL'\n",
                "    print(f\"[{status_icon}] Check {point:2d}: {res['check']} - {res['status']} ({res['evidence']})\")\n"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

for p in [Path("notebooks/01_DATASET_INSPECTION.ipynb"), Path("backend/notebooks/01_DATASET_INSPECTION.ipynb")]:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
print("Successfully generated 01_DATASET_INSPECTION.ipynb")
