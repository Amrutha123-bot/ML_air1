"""
Executes Phase 4 dataset inspection using aerosense_ml and outputs reports/dataset_inventory.csv.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from aerosense_ml.data.loaders import load_canonical_lakehouse
from aerosense_ml.data.inspection import inspect_city_data, generate_dataset_inventory
from aerosense_ml.evaluation.integrity import PredictionIntegrityAuditor

def main():
    print("Running AeroSense AI Phase 4 Dataset Inspection...")
    inspections = []
    
    for city, src in [
        ("delhi", "CPCB National Air Quality Monitoring Programme"),
        ("mumbai", "CPCB / MPCB Air Quality Monitoring Network"),
        ("vijayawada", "APPCB Ambient Air Monitoring Programme"),
    ]:
        try:
            df = load_canonical_lakehouse(city, base_dir="backend/data/canonical")
            insp = inspect_city_data(df, city_name=city, source_name=src)
            inspections.append(insp)
            print(f"Inspected {city}: {insp['total_rows']:,} rows, {insp['station_count']} stations, status: {insp['status']}")
        except Exception as e:
            print(f"Error inspecting {city}: {e}")
            
    inv_df = generate_dataset_inventory(inspections, output_path="reports/dataset_inventory.csv")
    print(f"\nSuccessfully generated reports/dataset_inventory.csv with {len(inv_df)} records:")
    print(inv_df[["City", "Total_Rows", "Stations", "Status"]].to_string())

    print("\nRunning 14-Point Integrity Audit:")
    auditor = PredictionIntegrityAuditor(backend_root=Path("backend"))
    audit_results = auditor.audit_all()
    for point, res in audit_results.items():
        print(f"  [{point:2d}] {res['check']}: {res['status']} ({res['evidence']})")

if __name__ == "__main__":
    main()
