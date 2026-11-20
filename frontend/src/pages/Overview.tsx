import React, { useEffect, useState } from 'react';
import { apiService } from '../api/client';
import type { HealthResponse, ForecastResponse, EarlyWarningItem } from '../types';
import { Card } from '../components/ui/Card';
import { StatCard, Badge } from '../components/ui/Badge';
import { getSeverityStyle } from '../utils/severity';
import {
  Activity,
  MapPin,
  AlertTriangle,
  Cpu,
  TrendingUp,
  Radio,
  ArrowRight,
  ShieldCheck,
  CheckCircle,
  Globe,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from 'recharts';

interface StationItem {
  station_id: string;
  name: string;
  latitude: number;
  longitude: number;
  site_type?: string;
  description?: string;
}

const CITIES_LIST = [
  { id: 'delhi', name: 'Delhi NCR', count: 12 },
  { id: 'mumbai', name: 'Mumbai', count: 8 },
  { id: 'bengaluru', name: 'Bengaluru', count: 5 },
  { id: 'kolkata', name: 'Kolkata', count: 5 },
  { id: 'hyderabad', name: 'Hyderabad', count: 4 },
  { id: 'chennai', name: 'Chennai', count: 4 },
  { id: 'ahmedabad', name: 'Ahmedabad', count: 4 },
  { id: 'pune', name: 'Pune', count: 4 },
  { id: 'jaipur', name: 'Jaipur', count: 3 },
  { id: 'lucknow', name: 'Lucknow', count: 4 },
  { id: 'vijayawada', name: 'Vijayawada', count: 2 },
];

export const Overview: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [selectedCity, setSelectedCity] = useState<string>('delhi');
  const [stations, setStations] = useState<StationItem[]>([]);
  const [selectedStationId, setSelectedStationId] = useState<string>('DEL_01');
  const [selectedStationName, setSelectedStationName] = useState<string>('Anand Vihar');
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [warnings, setWarnings] = useState<EarlyWarningItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Load initial system health
  useEffect(() => {
    apiService.getHealth()
      .then(setHealth)
      .catch((err) => setError(err?.message || 'Failed to connect to AeroSense backend API'));
  }, []);

  // When city changes, load stations for that city
  useEffect(() => {
    let isCancelled = false;
    const fetchCityData = async () => {
      try {
        setLoading(true);
        const cityStationsRes = await apiService.getCityStations(selectedCity);
        if (isCancelled) return;

        const cityStations: StationItem[] = (cityStationsRes.stations || []).map((s) => ({
          station_id: s.station_id,
          name: s.name,
          latitude: s.latitude,
          longitude: s.longitude,
          site_type: s.site_type,
          description: `${s.site_type || 'Monitoring Station'} - ${selectedCity.toUpperCase()} Network`,
        }));

        setStations(cityStations);

        if (cityStations.length > 0) {
          const first = cityStations[0];
          setSelectedStationId(first.station_id);
          setSelectedStationName(first.name);
          await loadStationTelemetry(selectedCity, first.station_id, first.name);
        }
        setError(null);
      } catch (err: any) {
        if (!isCancelled) {
          setError(err?.message || `Failed to connect to ${selectedCity} telemetry feed`);
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    };

    fetchCityData();
    return () => {
      isCancelled = true;
    };
  }, [selectedCity]);

  const loadStationTelemetry = async (city: string, stId: string, stName: string) => {
    try {
      if (city === 'delhi') {
        try {
          const [forecastData, warningData] = await Promise.all([
            apiService.getCitySTCNNForecast(city, stId, 0),
            apiService.getCityWarnings(city, stId, 0),
          ]);

          const mappedForecast: ForecastResponse = {
            station: stName,
            reference_datetime: forecastData.reference_datetime,
            current_pm25: forecastData.current_pm25,
            model_name: forecastData.model_name || `ST-CNN-GRU (${city.toUpperCase()})`,
            mode: forecastData.mode || 'city',
            predictions: forecastData.predictions || [],
            data_mode: forecastData.data_mode,
            units: forecastData.units || 'µg/m³',
          };

          setForecast(mappedForecast);
          setWarnings(warningData.warnings || []);
          return;
        } catch {
          // fallback to standard getForecast
        }
      }
      const [forecastData, warningData] = await Promise.all([
        apiService.getForecast(stName),
        apiService.getWarnings(stName),
      ]);
      setForecast(forecastData);
      setWarnings(warningData.warnings || []);
    } catch (err: any) {
      console.error('Error fetching station forecast:', err);
    }
  };

  const handleCityChange = (city: string) => {
    setSelectedCity(city);
    setForecast(null);
    setWarnings([]);
  };

  const handleStationChange = async (stId: string) => {
    const found = stations.find((s) => s.station_id === stId || s.name === stId);
    const resolvedId = found ? found.station_id : stId;
    const resolvedName = found ? found.name : stId;

    setSelectedStationId(resolvedId);
    setSelectedStationName(resolvedName);
    await loadStationTelemetry(selectedCity, resolvedId, resolvedName);
  };

  const forecastChartData = forecast?.predictions.map((p) => ({
    horizon: `+${p.horizon_hours}h`,
    pm25: p.predicted_pm25,
    tier: p.alert_tier,
  })) || [];

  const currentCityObj = CITIES_LIST.find((c) => c.id === selectedCity);
  const cityNameDisplay = currentCityObj ? currentCityObj.name : selectedCity;

  return (
    <div className="space-y-6">
      {/* System Status Alert if offline */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-5 h-5 text-rose-600" />
            <span><strong>Backend Connection Notice:</strong> {error}</span>
          </div>
          <button
            onClick={() => window.location.reload()}
            className="px-3 py-1 bg-rose-600 text-white rounded text-xs font-semibold hover:bg-rose-700 cursor-pointer"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* City Switcher Banner */}
      <div className="bg-white p-4 rounded-xl border border-lavender-200/90 shadow-xs flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-brand-50 border border-brand-200 flex items-center justify-center text-brand-700">
            <Globe className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-900">Active Metropolitan Area</h3>
            <p className="text-xs text-gray-500">Spatiotemporal telemetry & forecasting context</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <div className="hidden sm:flex items-center space-x-1.5">
            {CITIES_LIST.slice(0, 4).map((c) => (
              <button
                key={c.id}
                onClick={() => handleCityChange(c.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                  selectedCity === c.id
                    ? 'bg-brand-600 text-white shadow-xs'
                    : 'bg-lavender-50 text-gray-700 border border-lavender-200 hover:bg-lavender-100'
                }`}
              >
                {c.name}
              </button>
            ))}
          </div>

          <select
            value={selectedCity}
            onChange={(e) => handleCityChange(e.target.value)}
            className="h-8.5 bg-lavender-50/90 border border-lavender-300 text-xs font-bold rounded-lg px-2.5 py-1 text-brand-900 focus:outline-none focus:ring-2 focus:ring-brand-500 cursor-pointer shadow-xs"
          >
            {CITIES_LIST.map((c) => (
              <option key={c.id} value={c.id} className="bg-white text-gray-900 font-medium">
                {c.name} ({c.count} Stations)
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Top Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard
          title="Monitored Stations"
          value={stations.length ? `${stations.length} Active` : '12'}
          subtitle={`${cityNameDisplay} Grid`}
          icon={MapPin}
          colorClass="from-brand-600 to-brand-800"
        />
        <StatCard
          title="Active Model"
          value="ST-CNN-GRU (Delhi)"
          subtitle="Checkpoint: stcnn_gru_delhi.pt"
          icon={Cpu}
          colorClass="from-purple-600 to-indigo-800"
        />
        <StatCard
          title="Lead Time Horizons"
          value="6 Horizons"
          subtitle="1h, 6h, 12h, 24h, 48h, 72h ahead"
          icon={TrendingUp}
          colorClass="from-violet-600 to-purple-800"
        />
        <StatCard
          title="Regulatory Standard"
          value="CPCB NAAQS"
          subtitle="Tiered Threshold Alerting"
          icon={ShieldCheck}
          colorClass="from-emerald-600 to-teal-800"
        />
      </div>

      {/* Main Grid: Forecast Preview & Warning Highlights */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Forecast Trajectory Visualizer */}
        <Card
          className="lg:col-span-2"
          title={
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-brand-700" />
              <span>Multi-Horizon Forecast: {selectedStationName} ({cityNameDisplay})</span>
            </div>
          }
          subtitle={`Reference Date: ${forecast?.reference_datetime || 'Test Sequence Replay'} | Current PM2.5: ${forecast?.current_pm25?.toFixed(1) || '--'} µg/m³`}
          action={
            <select
              value={selectedStationId}
              onChange={(e) => handleStationChange(e.target.value)}
              className="bg-lavender-50 border border-lavender-200 text-brand-900 text-xs font-semibold rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-brand-500 cursor-pointer"
            >
              {stations.map((st) => (
                <option key={st.station_id} value={st.station_id}>
                  {st.name} ({st.station_id})
                </option>
              ))}
            </select>
          }
        >
          {loading ? (
            <div className="h-64 flex items-center justify-center text-sm text-gray-500">
              Loading {cityNameDisplay} station telemetry...
            </div>
          ) : (
            <div className="space-y-6">
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={forecastChartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0eef7" />
                    <XAxis dataKey="horizon" stroke="#71717a" fontSize={12} />
                    <YAxis stroke="#71717a" fontSize={12} unit=" µg/m³" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#ffffff',
                        borderColor: '#e2ddf0',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)',
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="pm25"
                      name="Predicted PM2.5 (ST-CNN-GRU)"
                      stroke="#6d28d9"
                      strokeWidth={3}
                      dot={{ r: 5, fill: '#6d28d9', stroke: '#ffffff', strokeWidth: 2 }}
                      activeDot={{ r: 7 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Horizon Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
                {forecast?.predictions.map((p) => {
                  const sev = getSeverityStyle(p.alert_tier, p.color_hex);
                  return (
                    <div
                      key={p.horizon_hours}
                      className="p-3.5 rounded-xl border flex flex-col justify-between shadow-2xs transition-all"
                      style={{
                        backgroundColor: sev.bg,
                        borderColor: sev.border,
                      }}
                    >
                      <div className="text-[11px] font-semibold" style={{ color: sev.text }}>
                        +{p.horizon_hours}h Ahead
                      </div>
                      <div className="text-xl font-bold font-mono my-1.5" style={{ color: sev.text }}>
                        {p.predicted_pm25.toFixed(1)} <span className="text-[10px] opacity-75 font-normal">µg/m³</span>
                      </div>
                      <Badge label={p.alert_tier} color={p.color_hex} />
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </Card>

        {/* Early Warning Feed Snapshot */}
        <Card
          title={
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5 text-amber-700" />
              <span>Active Warnings</span>
            </div>
          }
          subtitle={`${warnings.length} threshold alerts for ${selectedStationName}`}
          action={
            <Link
              to="/warnings"
              className="text-xs font-semibold text-brand-700 hover:text-brand-900 flex items-center space-x-1"
            >
              <span>View All</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          }
        >
          {warnings.length === 0 ? (
            <div className="p-8 text-center text-sm text-gray-500 space-y-2">
              <CheckCircle className="w-8 h-8 text-emerald-600 mx-auto" />
              <p className="font-semibold text-gray-700">Air Quality Acceptable</p>
              <p className="text-xs text-gray-400">No severe regulatory threshold breaches projected for {selectedStationName}.</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-[340px] overflow-y-auto pr-1">
              {warnings.map((w, idx) => {
                const sev = getSeverityStyle(w.severity_label, w.color_hex);
                return (
                  <div
                    key={idx}
                    className="p-3.5 rounded-xl border bg-white space-y-2.5 shadow-2xs transition-colors"
                    style={{ borderColor: sev.border }}
                  >
                    <div className="flex items-center justify-between">
                      <Badge label={w.severity_label} color={w.color_hex} />
                      <span className="text-xs font-mono font-medium text-gray-500">+{w.lead_time_hours}h lead time</span>
                    </div>
                    <p className="text-xs text-gray-800 font-medium leading-snug">{w.rationale}</p>
                    <div
                      className="text-[11px] p-2.5 rounded-lg border leading-relaxed font-medium"
                      style={{
                        backgroundColor: sev.bg,
                        color: sev.text,
                        borderColor: sev.border,
                      }}
                    >
                      <strong>Action:</strong> {w.recommended_action}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </Card>
      </div>

      {/* Network Station Grid Summary */}
      <Card
        title={`${cityNameDisplay} Monitoring Network Overview`}
        subtitle={`${stations.length} environmental reference stations with real geographical coordinates`}
        action={
          <Link
            to="/stations"
            className="text-xs font-semibold text-brand-700 hover:text-brand-900 flex items-center space-x-1"
          >
            <span>Network Explorer</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        }
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {stations.map((st) => (
            <button
              key={st.station_id}
              onClick={() => handleStationChange(st.station_id)}
              className={`p-4 rounded-xl border text-left transition-all cursor-pointer ${
                selectedStationId === st.station_id
                  ? 'border-brand-500 bg-lavender-50/80 shadow-xs ring-1 ring-brand-500'
                  : 'border-lavender-200/80 bg-white hover:border-brand-300 hover:bg-lavender-50/30'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-bold text-sm text-gray-900">{st.name}</span>
                <span className="text-[11px] font-mono text-gray-400">ID: {st.station_id}</span>
              </div>
              <p className="text-xs text-gray-500 line-clamp-2 leading-relaxed mb-2">
                {st.description}
              </p>
              <div className="text-[11px] font-mono text-brand-700 font-medium">
                {st.latitude.toFixed(4)}°N, {st.longitude.toFixed(4)}°E
              </div>
            </button>
          ))}
        </div>
      </Card>
    </div>
  );
};
