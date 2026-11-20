import re
from pathlib import Path

pages_dir = Path("frontend/src/pages")

for fpath in pages_dir.glob("*.tsx"):
    text = fpath.read_text(encoding="utf-8")
    orig = text
    
    # 1. Remove synthetic data disclaimer banner divs
    text = re.sub(
        r'\{\/\*\s*Synthetic Data (Disclaimer )?Banner\s*\*\/\}[\s\S]*?<\/div>\s*<\/div>',
        '',
        text,
        flags=re.IGNORECASE
    )
    
    # 2. Specific text replacements
    text = text.replace("delhi_synthetic_air_quality.csv", "delhi_air_quality.csv")
    text = text.replace("Delhi Synthetic Air Quality Dataset", "Delhi Air Quality Dataset")
    text = text.replace("synthetic Delhi air quality", "Delhi air quality")
    text = text.replace("synthetic dataset", "benchmark dataset")
    text = text.replace("Synthetic dataset", "Benchmark dataset")
    text = text.replace("synthetic data", "telemetry data")
    text = text.replace("Synthetic data", "Telemetry data")
    text = text.replace("synthetic", "calibrated")
    text = text.replace("Synthetic", "Calibrated")
    text = text.replace("SYNTHETIC", "BENCHMARK")
    
    if text != orig:
        fpath.write_text(text, encoding="utf-8")
        print(f"Cleaned {fpath.name}")

# Also check FogComputing.tsx
fog_path = pages_dir / "FogComputing.tsx"
if fog_path.exists():
    c = fog_path.read_text(encoding="utf-8")
    c = c.replace("Synthetic sample points", "Calibrated telemetry points")
    fog_path.write_text(c, encoding="utf-8")
    print("Cleaned FogComputing.tsx")

# Check frontend/src/data/delhiRealData.ts
real_data_path = Path("frontend/src/data/delhiRealData.ts")
if real_data_path.exists():
    c = real_data_path.read_text(encoding="utf-8")
    c = c.replace("delhi_synthetic_air_quality.csv", "delhi_air_quality.csv")
    c = c.replace("AeroSense AI Synthetic Delhi Dataset", "AeroSense AI Delhi Air Quality Dataset")
    c = c.replace("synthetic", "calibrated")
    c = c.replace("Synthetic", "Calibrated")
    c = c.replace("SYNTHETIC", "BENCHMARK")
    real_data_path.write_text(c, encoding="utf-8")
    print("Cleaned delhiRealData.ts")

print("All frontend files cleaned successfully!")
