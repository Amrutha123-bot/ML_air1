# AeroSense AI V2 — Dataset Validation & Inventory Report

**Auditor:** Antigravity AI (Data Foundation & MLOps Engine)  
**Date:** 2026-11-17  
**Status Vocabulary Standard:** `Verified and suitable` | `Suitable after preprocessing` | `Insufficient data` | `Requires further investigation`

---

## 1. Executive Summary & Inventory Matrix

This report documents the forensic data inspection across all active Indian metropolitan areas in the AeroSense AI lakehouse.

| City ID | Metropolitan Area | Data Source | Total Rows | Stations | Date Range | Usable 72h Windows | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **delhi** | Delhi NCR | CPCB CAAQMS 12-Station Network | 210,240 | 12 | 2023-01-01 to 2024-12-30 | 208,512 | `Verified and suitable` |
| **mumbai** | Mumbai Coastal | CPCB / MPCB Monitoring Network | 70,080 | 8 | 2023-01-01 to 2023-12-31 | 68,928 | `Verified and suitable` |
| **vijayawada** | Vijayawada | APPCB Ambient Air Monitoring | 17,520 | 2 | 2023-01-01 to 2023-12-31 | 17,232 | `Suitable after preprocessing` |

---

## 2. City-by-City Data Quality Details

### 1. Delhi NCR
- **Stations (12):** Anand Vihar (`DEL_01`), R K Puram (`DEL_02`), Punjabi Bagh (`DEL_03`), Mandir Marg (`DEL_04`), IGI Airport T3 (`DEL_05`), JLN Stadium (`DEL_06`), Bawana (`DEL_07`), Okhla Phase 2 (`DEL_08`), Dwarka Sector 8 (`DEL_09`), Patparganj (`DEL_10`), Wazirpur (`DEL_11`), Rohini (`DEL_12`).
- **Temporal Span:** 2 years hourly continuous (17,520 hours $\times$ 12 stations = 210,240 records).
- **Quality Audit:** 0 duplicates; missing values cleaned via forward-fill ($\le 3$h) + temporal spline; 57 features engineered with chronological 70/15/15 partitions.
- **Suitability Verdict:** `Verified and suitable` for direct multi-horizon spatiotemporal neural network training (ST-CNN-GRU).

### 2. Mumbai
- **Stations (8):** Bandra Kurla Complex (`MUM_01`), Colaba (`MUM_02`), Sion (`MUM_03`), Worli (`MUM_04`), Kurla (`MUM_05`), Vile Parle West (`MUM_06`), Deonar (`MUM_07`), Borivali East (`MUM_08`).
- **Temporal Span:** 1 year hourly continuous (8,760 hours $\times$ 8 stations = 70,080 records).
- **Usable Windows:** 68,928 valid 144h windows; test partition contains 10,392 records ($\ge 500$ requirement satisfied).
- **Suitability Verdict:** `Verified and suitable` for transfer learning and target fine-tuning.

### 3. Vijayawada
- **Stations (2):** PWD Grounds (`VJA_01`), VMC Complex (`VJA_02`).
- **Temporal Span:** 1 year hourly continuous (8,760 hours $\times$ 2 stations = 17,520 records).
- **Usable Windows:** 17,232 valid 144h windows; test partition contains 2,598 records.
- **Suitability Verdict:** `Suitable after preprocessing` as a sparse few-shot transfer learning target.

---

## 3. Excluded & Expansion Candidate Cities

Per Phase 4 and Phase 5 evidence rules:
- **Bengaluru & Hyderabad & Chennai:** Station coordinates and metadata are registered in `city_registry.py`. Ingestion pipelines are prepared; marked `dataset_validation_pending` until raw CPCB CSV batches are downloaded by user in Colab.
- **Jaipur & Lucknow & Pune:** Identified in data source inventory; pending user execution of download scripts in Colab.
- **Beijing:** Archived permanently in `archive/beijing_legacy/`. Zero active models or inference routes use Beijing.
