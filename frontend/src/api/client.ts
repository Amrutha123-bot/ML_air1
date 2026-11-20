import axios from 'axios';
import type {
  HealthResponse,
  StationListResponse,
  HistoricalObservation,
  ForecastResponse,
  WarningListResponse,
  WarningThresholdsResponse,
  ModelMetricsResponse,
} from '../types';

// ---- V2 Multi-City Types ----
export interface CityInfo {
  city_id: string;
  name: string;
  country: string;
  climate_zone: string;
  station_count: number;
  supported_models: string[];
  primary_model: string;
  data_mode: string;
  dataset_available: boolean;
}

export interface CityStationsResponse {
  city: string;
  total_stations: number;
  stations: Array<{
    station_id: string;
    name: string;
    latitude: number;
    longitude: number;
    site_type: string;
  }>;
}

export interface DatasetSummary {
  city_id: string;
  city_name: string;
  source: string;
  date_range: string;
  total_canonical_records: number;
  total_feature_rows: number;
  station_count: number;
  feature_count: number;
  preprocessing_status: string;
  supported_models: string[];
}

export interface EvaluationSummary {
  cities: string[];
  delhi_summary?: Record<string, any>;
  [key: string]: any;
}

export interface CityComparison {
  city_id: string;
  name: string;
  stations: number;
  climate_zone: string;
  mean_pm25: number;
  max_pm25: number;
  dispersion_regime: string;
  primary_model: string;
}

export interface PollutantTrend {
  datetime: string;
  'PM2.5': number;
  PM10: number;
  SO2: number;
  NO2: number;
  CO: number;
  O3: number;
  TEMP: number;
  WSPM: number;
}

export interface CorrelationItem {
  pollutant: string;
  [key: string]: number | string;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  getHealth: async (): Promise<HealthResponse> => {
    const res = await apiClient.get<HealthResponse>('/health');
    return res.data;
  },

  getStations: async (): Promise<StationListResponse> => {
    const res = await apiClient.get<StationListResponse>('/stations');
    return res.data;
  },

  getStationHistory: async (
    stationName: string,
    limit: number = 24,
    offset: number = 0
  ): Promise<HistoricalObservation[]> => {
    const res = await apiClient.get<HistoricalObservation[]>(
      `/stations/${encodeURIComponent(stationName)}/history`,
      { params: { limit, offset } }
    );
    return res.data;
  },

  getForecast: async (
    stationName: string,
    model?: string,
    horizon?: number,
    offset: number = 0
  ): Promise<ForecastResponse> => {
    const params: Record<string, any> = { offset };
    if (model) params.model = model;
    if (horizon) params.horizon = horizon;
    const res = await apiClient.get<ForecastResponse>(
      `/forecasts/${encodeURIComponent(stationName)}`,
      { params }
    );
    return res.data;
  },

  getWarnings: async (
    stationName: string,
    model?: string,
    offset: number = 0,
    city?: string
  ): Promise<WarningListResponse> => {
    const params: Record<string, any> = { offset };
    if (model) params.model = model;
    if (city) params.city = city;
    const res = await apiClient.get<WarningListResponse>(
      `/warnings/${encodeURIComponent(stationName)}`,
      { params }
    );
    return res.data;
  },

  getCityWarnings: async (
    city: string,
    stationId: string,
    offset: number = 0
  ): Promise<WarningListResponse> => {
    const res = await apiClient.get<WarningListResponse>(
      `/warnings/city/${encodeURIComponent(city)}/${encodeURIComponent(stationId)}`,
      { params: { offset } }
    );
    return res.data;
  },

  getWarningThresholds: async (): Promise<WarningThresholdsResponse> => {
    const res = await apiClient.get<WarningThresholdsResponse>('/warnings/thresholds');
    return res.data;
  },

  getMetrics: async (): Promise<ModelMetricsResponse> => {
    const res = await apiClient.get<ModelMetricsResponse>('/metrics');
    return res.data;
  },

  getCitySTCNNForecast: async (
    city: string,
    stationId: string,
    offset: number = 0
  ): Promise<any> => {
    const res = await apiClient.get(
      `/forecasts/city/${encodeURIComponent(city)}/${encodeURIComponent(stationId)}/stcnn`,
      { params: { offset } }
    );
    return res.data;
  },

  getCities: async (): Promise<any[]> => {
    const res = await apiClient.get('/cities');
    return res.data;
  },

  getCity: async (city: string): Promise<any> => {
    const res = await apiClient.get(`/cities/${encodeURIComponent(city)}`);
    return res.data;
  },

  getCityStations: async (city: string): Promise<any> => {
    const res = await apiClient.get(`/cities/${encodeURIComponent(city)}/stations`);
    return res.data;
  },

  getDatasets: async (): Promise<any[]> => {
    const res = await apiClient.get('/datasets');
    return res.data;
  },

  getDataset: async (city: string): Promise<any> => {
    const res = await apiClient.get(`/datasets/${encodeURIComponent(city)}`);
    return res.data;
  },

  getModelsRegistry: async (includeExperimental: boolean = true): Promise<any[]> => {
    const res = await apiClient.get('/models', {
      params: { include_experimental: includeExperimental },
    });
    return res.data;
  },

  getModelDetails: async (modelName: string): Promise<any> => {
    const res = await apiClient.get(`/models/${encodeURIComponent(modelName)}`);
    return res.data;
  },

  getEvaluationSummaries: async (): Promise<any> => {
    const res = await apiClient.get('/evaluation');
    return res.data;
  },

  getCityEvaluation: async (city: string): Promise<any> => {
    const res = await apiClient.get(`/evaluation/${encodeURIComponent(city)}`);
    return res.data;
  },

  getStationTrends: async (
    city: string,
    station: string,
    limit: number = 24
  ): Promise<any[]> => {
    const res = await apiClient.get(
      `/analytics/trends/${encodeURIComponent(city)}/${encodeURIComponent(station)}`,
      { params: { limit } }
    );
    return res.data;
  },

  getCorrelation: async (city: string): Promise<any[]> => {
    const res = await apiClient.get(`/analytics/correlation/${encodeURIComponent(city)}`);
    return res.data;
  },

  getComparison: async (): Promise<any[]> => {
    const res = await apiClient.get('/analytics/comparison');
    return res.data;
  },
};

// ── Standalone convenience exports (used by page components) ────────────────

export const getForecast = (
  stationName: string = 'Anand Vihar',
  model?: string,
  horizon?: number,
  offset: number = 0
) => apiService.getForecast(stationName, model, horizon, offset);

export const getWarnings = (
  stationName: string = 'Anand Vihar',
  model?: string,
  offset: number = 0
) => apiService.getWarnings(stationName, model, offset);

export const getMetrics = () => apiService.getMetrics();

export const getLeaderboard = async (): Promise<any> => {
  const res = await apiClient.get('/metrics/leaderboard');
  return res.data;
};

export const getConfusionMatrix = async (): Promise<any> => {
  const res = await apiClient.get('/metrics/confusion');
  return res.data;
};

export const getFractionResults = async (): Promise<any> => {
  const res = await apiClient.get('/metrics/fractions');
  return res.data;
};

export const getEDA = async (): Promise<any> => {
  const res = await apiClient.get('/analytics/eda');
  return res.data;
};

export const getStation = (stationName: string = 'Anand Vihar', limit: number = 24) =>
  apiService.getStationHistory(stationName, limit);

export const getTrends = async (city: string = 'delhi', station: string = 'Anand Vihar', limit: number = 24): Promise<any[]> => {
  const res = await apiClient.get(`/analytics/trends/${encodeURIComponent(city)}/${encodeURIComponent(station)}`, { params: { limit } });
  return res.data;
};



