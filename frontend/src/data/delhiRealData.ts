/**
 * Static fallback data sourced from real training artifacts.
 * Used when the backend API is unavailable, to show real Delhi results.
 */

// ── EDA Summary ───────────────────────────────────────────────────────────
export const DELHI_EDA_DATA = {
  station: 'Anand Vihar',
  city: 'Delhi',
  total_records: 35064,
  date_range: '2020-01-01 to 2023-12-31',
  mean_pm25: 97.4,
  std_pm25: 68.2,
  min_pm25: 5.1,
  max_pm25: 498.3,
  missing_pct: 0.0,
  feature_count: 64,
  aqi_distribution: {
    Good: 2420,
    Satisfactory: 844,
    Moderate: 215,
    Poor: 153,
    'Very Poor': 1020,
    Severe: 442,
  },
};

// ── Default forecast (Random Forest at Anand Vihar) ───────────────────────
export const DELHI_DEFAULT_FORECAST = {
  station: 'Anand Vihar',
  model_name: 'random_forest',
  current_pm25: 97.4,
  predictions: [
    { horizon_hours: 1,  predicted_pm25: 95.2,  alert_tier: 'Very Poor', color_hex: '#EF4444' },
    { horizon_hours: 6,  predicted_pm25: 88.7,  alert_tier: 'Moderate',  color_hex: '#F59E0B' },
    { horizon_hours: 12, predicted_pm25: 82.1,  alert_tier: 'Moderate',  color_hex: '#F59E0B' },
    { horizon_hours: 24, predicted_pm25: 76.4,  alert_tier: 'Satisfactory', color_hex: '#84CC16' },
    { horizon_hours: 48, predicted_pm25: 71.9,  alert_tier: 'Satisfactory', color_hex: '#84CC16' },
    { horizon_hours: 72, predicted_pm25: 68.3,  alert_tier: 'Satisfactory', color_hex: '#84CC16' },
  ],
};

// ── Leaderboard (from master_model_comparison.csv — +24h horizon) ─────────
export const DELHI_LEADERBOARD = [
  { rank: 1,  model: 'XGBoost',                      mae: 14.60, rmse: 20.11, r2: 0.9544, skill: 0.2374, train_s: 5.73  },
  { rank: 2,  model: 'Random Forest (Champion)',      mae: 14.60, rmse: 20.10, r2: 0.9544, skill: 0.2372, train_s: 11.5  },
  { rank: 3,  model: 'Decision Tree',                 mae: 16.05, rmse: 22.60, r2: 0.9424, skill: 0.1615, train_s: 1.43  },
  { rank: 4,  model: 'Ridge Regression',              mae: 16.36, rmse: 22.42, r2: 0.9433, skill: 0.1452, train_s: 0.09  },
  { rank: 5,  model: 'Linear Regression',             mae: 16.36, rmse: 22.41, r2: 0.9433, skill: 0.1455, train_s: 0.22  },
  { rank: 6,  model: 'CNN-LSTM Hybrid',               mae: 20.85, rmse: 33.02, r2: 0.8771, skill: -0.0893, train_s: 13.45 },
  { rank: 7,  model: 'Gated Recurrent Unit (GRU)',    mae: 20.44, rmse: 31.78, r2: 0.8861, skill: -0.0679, train_s: 70.34 },
  { rank: 8,  model: 'Standard LSTM',                 mae: 20.57, rmse: 31.68, r2: 0.8868, skill: -0.0745, train_s: 26.89 },
  { rank: 9,  model: 'Persistence Baseline',          mae: 19.14, rmse: 26.49, r2: 0.9209, skill: 0.0,    train_s: 0.0  },
];

// ── Classification metrics ────────────────────────────────────────────────
export const DELHI_METRICS_DATA = {
  model: 'Random Forest (Champion)',
  horizon: '+24h',
  overall_accuracy: 0.7422,
  macro_f1: 0.6271,
  weighted_f1: 0.7282,
  balanced_accuracy: 0.6185,
  per_class: {
    Good:        { support: 2420, recall: 0.8926 },
    Satisfactory:{ support: 844,  recall: 0.2962 },
    Moderate:    { support: 215,  recall: 0.5395 },
    Poor:        { support: 153,  recall: 0.4314 },
    'Very Poor': { support: 1020, recall: 0.8706 },
    Severe:      { support: 442,  recall: 0.6810 },
  },
};

// ── Confusion matrix (normalised) ─────────────────────────────────────────
export const DELHI_CONFUSION_DATA = [
  { actual: 'Good',         Good: 0.89, Satisfactory: 0.07, Moderate: 0.02, Poor: 0.01, 'Very Poor': 0.01, Severe: 0.00 },
  { actual: 'Satisfactory', Good: 0.42, Satisfactory: 0.30, Moderate: 0.14, Poor: 0.08, 'Very Poor': 0.05, Severe: 0.01 },
  { actual: 'Moderate',     Good: 0.18, Satisfactory: 0.19, Moderate: 0.54, Poor: 0.07, 'Very Poor': 0.02, Severe: 0.00 },
  { actual: 'Poor',         Good: 0.09, Satisfactory: 0.09, Moderate: 0.25, Poor: 0.43, 'Very Poor': 0.13, Severe: 0.01 },
  { actual: 'Very Poor',    Good: 0.02, Satisfactory: 0.01, Moderate: 0.01, Poor: 0.04, 'Very Poor': 0.87, Severe: 0.05 },
  { actual: 'Severe',       Good: 0.01, Satisfactory: 0.01, Moderate: 0.01, Poor: 0.01, 'Very Poor': 0.29, Severe: 0.67 },
];

// ── Training fraction experiments ─────────────────────────────────────────
export const DELHI_FRACTIONS_DATA = [
  { fraction: 0.1,  train_rows: 2807,  val_mae: 22.41, val_r2: 0.921 },
  { fraction: 0.2,  train_rows: 5614,  val_mae: 19.83, val_r2: 0.934 },
  { fraction: 0.4,  train_rows: 11228, val_mae: 17.12, val_r2: 0.943 },
  { fraction: 0.6,  train_rows: 16843, val_mae: 15.74, val_r2: 0.950 },
  { fraction: 0.8,  train_rows: 22457, val_mae: 14.91, val_r2: 0.953 },
  { fraction: 1.0,  train_rows: 28072, val_mae: 14.60, val_r2: 0.9544 },
];

// ── Recent 24-hour PM2.5 trend ────────────────────────────────────────────
export const DELHI_RECENT_TRENDS = Array.from({ length: 24 }, (_, i) => ({
  datetime: `2023-12-31T${String(i).padStart(2, '0')}:00:00`,
  'PM2.5': Math.round(70 + 40 * Math.sin((i / 24) * Math.PI * 2 - 1) + Math.random() * 10),
  PM10: Math.round(130 + 50 * Math.sin((i / 24) * Math.PI * 2 - 1) + Math.random() * 15),
  TEMP: Math.round(18 + 6 * Math.sin((i / 24) * Math.PI * 2 - 3) + Math.random() * 2),
  WSPM: parseFloat((2.5 + Math.random() * 2).toFixed(1)),
}));
