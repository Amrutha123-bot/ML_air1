export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
  active_model: string;
  available_models: string[];
  device?: string;
  available_cities_count?: number;
  available_cities?: string[];
  dataset_availability?: Record<string, boolean>;
  data_mode?: string;
  dataset_date_range?: string;
  real_time_ingestion?: boolean;
}

export interface StationInfo {
  station_id: number;
  name: string;
  latitude: number;
  longitude: number;
  description: string;
}

export interface StationListResponse {
  total_stations: number;
  stations: StationInfo[];
}

export interface HistoricalObservation {
  datetime: string;
  station: string;
  "PM2.5": number;
  PM10?: number | null;
  SO2?: number | null;
  NO2?: number | null;
  CO?: number | null;
  O3?: number | null;
  TEMP?: number | null;
  PRES?: number | null;
  DEWP?: number | null;
  RAIN?: number | null;
  WSPM?: number | null;
  wd_sin?: number | null;
  wd_cos?: number | null;
}

export interface ForecastHorizonPrediction {
  horizon_hours: number;
  predicted_pm25: number;
  alert_tier: string;
  color_hex: string;
}

export interface ForecastResponse {
  station: string;
  reference_datetime: string;
  model_name: string;
  mode: string;
  current_pm25: number;
  predictions: ForecastHorizonPrediction[];
  artifact_path?: string;
  artifact_size_bytes?: number;
  input_observation_datetime?: string;
  prediction_generation_datetime?: string;
  data_source?: string;
  data_mode?: string;
  units?: string;
  feature_count?: number;
  sequence_length?: number;
}

export interface EarlyWarningItem {
  station: string;
  lead_time_hours: number;
  predicted_pm25: number;
  warning_category: string;
  alert_code: string;
  severity_label: string;
  color_hex: string;
  trend_type: string;
  is_acute_spike: boolean;
  rationale: string;
  recommended_action: string;
  reference_datetime: string;
}

export interface WarningListResponse {
  station: string;
  total_warnings: number;
  highest_severity: string;
  warnings: EarlyWarningItem[];
}

export interface WarningThresholdsResponse {
  standard: string;
  averaging_period: string;
  tiers: Record<string, {
    min: number;
    max: number;
    label: string;
    color: string;
    health_implications: string;
  }>;
}

export interface MetricSummaryItem {
  model: string;
  overall_MAE: number;
  overall_RMSE: number;
  overall_R2: number;
  warning_F1: number;
  skill_score_vs_baseline: number;
}

export interface ModelMetricsResponse {
  benchmark_dataset: string;
  summary: MetricSummaryItem[];
  details?: Record<string, any>;
}
