# Frontend Integration Handover Documentation
## Beijing Multi-Station Air Quality Forecasting & Early-Warning System

**Target Audience:** Frontend Engineering Team  
**Backend Version:** `1.0.0` (FastAPI REST backend)  
**API Specification:** OpenAPI 3.1 (`/docs` for Swagger UI, `/redoc` for ReDoc)  
**Historical Period Covered:** March 1, 2013 – February 28, 2017 (UCI Beijing Multi-Site Air Quality Dataset)

---

## 1. API URLs and Environments

| Environment | Base URL | Interactive Docs | Description |
|-------------|----------|------------------|-------------|
| **Local Development** | `http://localhost:8000/api/v1` | `http://localhost:8000/docs` | Standard local development server |
| **Production / Cloud** | `https://beijing-air-quality-api.onrender.com/api/v1` | `https://beijing-air-quality-api.onrender.com/docs` | Production deployment URL |

### CORS Configuration
The backend allows cross-origin requests from all origins (`*`) by default, including:
- `http://localhost:3000` (React / Next.js default)
- `http://localhost:5173` (Vite default)
- `http://localhost:8080` (Vue default)
- All production origins via `CORS_ORIGINS` environment variable.

---

## 2. Monitoring Stations Reference Data

All 12 monitoring stations are included with their official geographical coordinates:

```json
[
  {"station_id": 0, "name": "Aotizhongxin", "latitude": 39.9823, "longitude": 116.3975, "description": "Olympic Sports Center, North Chaoyang (Urban residential & athletic area)"},
  {"station_id": 1, "name": "Changping", "latitude": 40.2175, "longitude": 116.2310, "description": "Changping suburban district, Northwest Beijing (Mountain-basin boundary)"},
  {"station_id": 2, "name": "Dingling", "latitude": 40.2922, "longitude": 116.2204, "description": "Ming Tombs ecological background station, Northern clean reference"},
  {"station_id": 3, "name": "Dongsi", "latitude": 39.9289, "longitude": 116.4173, "description": "Dongcheng inner city, Central Beijing (Dense commercial & traffic corridor)"},
  {"station_id": 4, "name": "Guanyuan", "latitude": 39.9289, "longitude": 116.3390, "description": "Xicheng central residential district (Governmental & educational center)"},
  {"station_id": 5, "name": "Gucheng", "latitude": 39.9144, "longitude": 116.1844, "description": "Shijingshan industrial heritage zone, Western Beijing (Near Western Hills)"},
  {"station_id": 6, "name": "Huairou", "latitude": 40.3756, "longitude": 116.6321, "description": "Huairou northern mountain resort & reservoir basin (Regional background)"},
  {"station_id": 7, "name": "Nongzhanguan", "latitude": 39.9373, "longitude": 116.4607, "description": "National Agricultural Exhibition Center, East Chaoyang (Embassy & commercial hub)"},
  {"station_id": 8, "name": "Shunyi", "latitude": 40.1270, "longitude": 116.6553, "description": "Northeastern industrial/airport logistics district (Capital International Airport area)"},
  {"station_id": 9, "name": "Tiantan", "latitude": 39.8863, "longitude": 116.4072, "description": "Temple of Heaven Park, South Dongcheng (Historical cultural preserve)"},
  {"station_id": 10, "name": "Wanliu", "latitude": 39.9876, "longitude": 116.2996, "description": "Haidian high-tech and university district, Northwest urban perimeter"},
  {"station_id": 11, "name": "Wanshouxigong", "latitude": 39.8781, "longitude": 116.3522, "description": "Wanshou West Palace, Xicheng Southern residential center"}
]
```

---

## 3. Early-Warning Alert Tiers and Color Palette

Use these exact hex codes and badges in your frontend components:

| Category Key | Alert Code | PM2.5 Range (μg/m³) | Color Hex | Label | Actionable Public Advisory |
|--------------|------------|---------------------|-----------|-------|----------------------------|
| `good` | `GREEN` | 0 – 35 | `#00E400` | Good (Grade I) | Air quality is satisfactory; outdoor activities are safe. |
| `moderate` | `YELLOW_LOW` | 35 – 75 | `#FFFF00` | Moderate (Grade II) | Acceptable air quality; unusually sensitive individuals limit prolonged exertion. |
| `unhealthy_sg` | `BLUE` | 75 – 115 | `#FF7E00` | Sensitive Groups | Children & elderly reduce prolonged outdoor exertion. |
| `unhealthy` | `YELLOW` | 115 – 150 | `#FF0000` | Unhealthy (Yellow) | General public reduce outdoor exertion; sensitive groups stay indoors. |
| `very_unhealthy` | `ORANGE` | 150 – 250 | `#8F3F97` | Very Unhealthy (Orange) | Outdoor activities suspended for schools; vulnerable groups stay indoors. |
| `hazardous` | `RED` | > 250 | `#7E0023` | Hazardous (Red) | Emergency health warning: entire population affected. Heavy outdoor restrictions. |

---

## 4. Endpoints Specification

### 4.1 System Health
- **Route:** `GET /api/v1/health`
- **Description:** Verifies service health, active model in memory, and list of loaded models.
- **Example Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-11-12T12:45:00.000Z",
  "active_model": "stcnn_gru",
  "available_models": ["persistence", "random_forest", "xgboost", "lstm", "cnn_lstm", "stcnn_gru"]
}
```

### 4.2 List Monitoring Stations
- **Route:** `GET /api/v1/stations`
- **Description:** Returns all 12 stations with lat/lon and descriptions for map rendering (e.g. Leaflet / Mapbox).
- **Example Response:**
```json
{
  "total_stations": 12,
  "stations": [
    {
      "station_id": 3,
      "name": "Dongsi",
      "latitude": 39.9289,
      "longitude": 116.4173,
      "description": "Dongcheng inner city, Central Beijing (Dense commercial & traffic corridor)"
    }
  ]
}
```

### 4.3 Station Historical Observations (Replay)
- **Route:** `GET /api/v1/stations/{station_name}/history`
- **Query Parameters:**
  - `limit` (int, default 24, max 168): Number of historical hours.
  - `offset` (int, default 0): Hours back from the test period end.
- **Example Response:**
```json
[
  {
    "datetime": "2017-02-28 23:00:00",
    "station": "Dongsi",
    "PM2.5": 82.0,
    "PM10": 110.0,
    "SO2": 15.0,
    "NO2": 45.0,
    "CO": 1200.0,
    "O3": 25.0,
    "TEMP": 6.5,
    "PRES": 1018.2,
    "DEWP": -11.0,
    "RAIN": 0.0,
    "WSPM": 1.8
  }
]
```

### 4.4 Multi-Horizon Forecasts
- **Route:** `GET /api/v1/forecasts/{station_name}`
- **Query Parameters:**
  - `model` (string, optional): One of `stcnn_gru`, `xgboost`, `random_forest`, `lstm`, `cnn_lstm`, `persistence`. (Defaults to best model).
  - `horizon` (int, optional): Specific lead time (`1`, `6`, `12`, `24`, `48`, `72`). Omit to get all 6 horizons simultaneously.
  - `offset` (int, optional, default 0): Sequence step back.
- **Example Response:**
```json
{
  "station": "Dongsi",
  "reference_datetime": "2017-02-28 23:00:00",
  "model_name": "stcnn_gru",
  "mode": "historical_simulation",
  "current_pm25": 82.0,
  "predictions": [
    {"horizon_hours": 1, "predicted_pm25": 86.4, "alert_tier": "Unhealthy for Sensitive Groups", "color_hex": "#FF7E00"},
    {"horizon_hours": 6, "predicted_pm25": 105.1, "alert_tier": "Unhealthy for Sensitive Groups", "color_hex": "#FF7E00"},
    {"horizon_hours": 12, "predicted_pm25": 138.7, "alert_tier": "Unhealthy (Yellow Alert)", "color_hex": "#FF0000"},
    {"horizon_hours": 24, "predicted_pm25": 162.3, "alert_tier": "Very Unhealthy (Orange Alert)", "color_hex": "#8F3F97"},
    {"horizon_hours": 48, "predicted_pm25": 141.0, "alert_tier": "Unhealthy (Yellow Alert)", "color_hex": "#FF0000"},
    {"horizon_hours": 72, "predicted_pm25": 65.2, "alert_tier": "Moderate (Grade II)", "color_hex": "#FFFF00"}
  ]
}
```

### 4.5 Early Warnings & Explanations
- **Route:** `GET /api/v1/warnings/{station_name}`
- **Query Parameters:**
  - `model` (string, optional): Forecasting model to evaluate.
  - `offset` (int, optional): Sequence step back.
- **Example Response:**
```json
{
  "station": "Dongsi",
  "total_warnings": 6,
  "highest_severity": "very_unhealthy",
  "warnings": [
    {
      "station": "Dongsi",
      "lead_time_hours": 24,
      "predicted_pm25": 162.3,
      "warning_category": "very_unhealthy",
      "alert_code": "ORANGE",
      "severity_label": "Very Unhealthy (Orange Alert)",
      "color_hex": "#8F3F97",
      "trend_type": "gradual_accumulation",
      "is_acute_spike": false,
      "rationale": "Warning at +24h lead time: Predicted PM2.5 of 162.3 ug/m^3 exceeds regulatory alert limit for Very Unhealthy (Orange Alert). Trend: gradual accumulation.",
      "recommended_action": "Outdoor activities should be suspended for schools; vulnerable groups stay indoors.",
      "reference_datetime": "2017-02-28 23:00:00"
    }
  ]
}
```

### 4.6 Model Evaluation & Scientific Benchmark
- **Route:** `GET /api/v1/metrics`
- **Description:** Returns model performance on untouched test set (MAE, RMSE, R², Warning F1, and Skill Score vs Persistence).
- **Example Response:**
```json
{
  "benchmark_dataset": "UCI Beijing Multi-Site Air Quality Dataset (Untouched Test Split)",
  "summary": [
    {
      "model": "stcnn_gru",
      "overall_MAE": 34.2,
      "overall_RMSE": 52.8,
      "overall_R2": 0.68,
      "warning_F1": 0.81,
      "skill_score_vs_baseline": 0.38
    },
    {
      "model": "xgboost",
      "overall_MAE": 35.8,
      "overall_RMSE": 54.1,
      "overall_R2": 0.65,
      "warning_F1": 0.79,
      "skill_score_vs_baseline": 0.36
    },
    {
      "model": "persistence",
      "overall_MAE": 57.5,
      "overall_RMSE": 85.6,
      "overall_R2": 0.12,
      "warning_F1": 0.58,
      "skill_score_vs_baseline": 0.0
    }
  ]
}
```

---

## 5. TypeScript Interfaces for Frontend

Copy-paste these definitions into `src/types/airQuality.ts`:

```typescript
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

export interface ForecastHorizonPrediction {
  horizon_hours: 1 | 6 | 12 | 24 | 48 | 72;
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
}

export interface EarlyWarningItem {
  station: string;
  lead_time_hours: number;
  predicted_pm25: number;
  warning_category: "good" | "moderate" | "unhealthy_sg" | "unhealthy" | "very_unhealthy" | "hazardous";
  alert_code: "GREEN" | "YELLOW_LOW" | "BLUE" | "YELLOW" | "ORANGE" | "RED";
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

export interface MetricSummaryItem {
  model: string;
  overall_MAE: number;
  overall_RMSE: number;
  overall_R2: number;
  warning_F1: number;
  skill_score_vs_baseline: number;
}
```

---

## 6. Ready-to-Use JavaScript / TypeScript API Client

```typescript
const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export async function fetchStations(): Promise<StationInfo[]> {
  const res = await fetch(`${BASE_URL}/stations`);
  if (!res.ok) throw new Error(`Failed to fetch stations: ${res.statusText}`);
  const data = await res.json();
  return data.stations;
}

export async function fetchStationForecast(
  station: string,
  model?: string,
  horizon?: number
): Promise<ForecastResponse> {
  const params = new URLSearchParams();
  if (model) params.append("model", model);
  if (horizon) params.append("horizon", horizon.toString());

  const res = await fetch(`${BASE_URL}/forecasts/${encodeURIComponent(station)}?${params}`);
  if (!res.ok) throw new Error(`Forecast request failed: ${res.statusText}`);
  return res.json();
}

export async function fetchStationWarnings(
  station: string,
  model?: string
): Promise<WarningListResponse> {
  const params = model ? `?model=${encodeURIComponent(model)}` : "";
  const res = await fetch(`${BASE_URL}/warnings/${encodeURIComponent(station)}${params}`);
  if (!res.ok) throw new Error(`Warnings request failed: ${res.statusText}`);
  return res.json();
}

export async function fetchMetrics(): Promise<MetricSummaryItem[]> {
  const res = await fetch(`${BASE_URL}/metrics`);
  if (!res.ok) throw new Error(`Metrics request failed: ${res.statusText}`);
  const data = await res.json();
  return data.summary;
}
```

---

## 7. Error Formats and Status Codes

The API returns standard RFC 7807 compliant JSON errors:

```json
{
  "detail": "Station 'InvalidName' not found"
}
```

| HTTP Status | Meaning | Solution for Frontend |
|-------------|---------|-----------------------|
| `200 OK` | Request succeeded | Render charts / components |
| `400 Bad Request` | Invalid query parameter (e.g. limit > 168) | Validate inputs before sending |
| `404 Not Found` | Station name doesn't exist | Verify station name matches `/stations` |
| `500 Server Error` | Model inference fault | Display friendly retry toast |
