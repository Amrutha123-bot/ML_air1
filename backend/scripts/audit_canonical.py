import pandas as pd
from pathlib import Path

for city in ['delhi', 'mumbai', 'vijayawada']:
    p = Path(f'backend/data/canonical/{city}/observations.parquet')
    if p.exists():
        df = pd.read_parquet(p)
        print(f"=== {city.upper()} ===")
        print(f"Rows: {len(df)}")
        print(f"Stations: {df['station_id'].nunique()} -> {list(df['station_id'].unique())}")
        ts = df['timestamp_local'] if 'timestamp_local' in df.columns else df['timestamp_utc']
        print(f"Date range: {ts.min()} to {ts.max()}")
        print(f"Missing PM2.5: {df['PM2.5'].isna().mean():.2%}")
        
        # Usable 144h windows (24h input + 72h max horizon + 48h purge)
        # Check per station contiguous hours
        min_windows = 0
        for st in df['station_id'].unique():
            st_df = df[df['station_id'] == st]
            valid_hours = len(st_df[st_df['PM2.5'].notna()])
            windows = max(0, valid_hours - 144)
            min_windows += windows
        print(f"Total usable 144h windows across stations: {min_windows}")
        
        # Check features file
        feat_p = Path(f'backend/data/features/{city}/test_features.parquet')
        if feat_p.exists():
            fdf = pd.read_parquet(feat_p)
            print(f"Test feature rows: {len(fdf)}")
        else:
            print("Test features: NOT FOUND")
        print()
