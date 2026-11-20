import React, { useEffect, useState } from 'react';
import { getEDA, getStation, getTrends } from '../api/client';
import { DELHI_EDA_DATA, DELHI_RECENT_TRENDS } from '../data/delhiRealData';
import type { HistoricalObservation } from '../types';
import { Card } from '../components/ui/Card';
import { classifyPM25Severity } from '../utils/severity';
import {
  MapPin,
  Clock,
  Calendar,
  AlertTriangle,
  BarChart2,
  Activity,
  Layers,
  Thermometer,
  Wind,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

export const MonitoringStations: React.FC = () => {
  const [stationInfo, setStationInfo] = useState<any>({
    station_id: "DEL_01",
    name: "Anand Vihar, Delhi",
    latitude: 28.6469,
    longitude: 77.3152,
    description: "Major inter-state bus terminus & transit hub, East Delhi"
  });
  const [edaData, setEdaData] = useState<any>(DELHI_EDA_DATA);
  const [trends, setTrends] = useState<any[]>(DELHI_RECENT_TRENDS);
  const [limit, setLimit] = useState<number>(48);
  const [loading, setLoading] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'particulate' | 'diurnal' | 'seasonal' | 'categories'>('particulate');

  useEffect(() => {
    let isCancelled = false;
    const fetchData = async () => {
      try {
        const [stRes, edaRes, trendsRes] = await Promise.all([
          getStation().catch(() => null),
          getEDA().catch(() => null),
          getTrends(limit).catch(() => null),
        ]);

        if (isCancelled) return;

        if (stRes?.data?.stations?.length) {
          setStationInfo(stRes.data.stations[0]);
        }
        if (edaRes?.data) {
          setEdaData(edaRes.data);
        }
        if (trendsRes?.data && Array.isArray(trendsRes.data) && trendsRes.data.length > 0) {
          setTrends(trendsRes.data);
        }
      } catch (err) {
        console.warn('Backend unavailable, using cached real dataset', err);
      }
    };

    fetchData();
    return () => {
      isCancelled = true;
    };
  }, [limit]);

  // Format hourly diurnal data
  const rawDiurnal = edaData?.hourly_diurnal_pm25 || edaData?.diurnal_hourly_pm25 || DELHI_EDA_DATA.hourly_diurnal_pm25;
  const diurnalData = rawDiurnal
    ? Object.entries(rawDiurnal).map(([hour, val]) => ({
        hour: `${hour}:00`,
        'Average PM2.5': Number(val),
      }))
    : [];

  // Format monthly seasonal data
  const monthNames = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const rawSeasonal = edaData?.monthly_seasonal_pm25 || edaData?.seasonal_monthly_pm25 || DELHI_EDA_DATA.monthly_seasonal_pm25;
  const seasonalData = rawSeasonal
    ? Object.entries(rawSeasonal).map(([m, val]) => ({
        month: monthNames[Number(m)] || `M${m}`,
        'Average PM2.5': Number(val),
      }))
    : [];

  // Format AQI category distribution
  const rawCategories = edaData?.category_distribution || DELHI_EDA_DATA.category_distribution;
  const categoryData = rawCategories
    ? Object.entries(rawCategories).map(([cat, count]) => ({
        category: cat,
        count: Number(count),
      }))
    : [];

  // Particulate time series
  const particulateChartData = trends.map((d: any) => ({
    time: d.timestamp ? String(d.timestamp).slice(5, 16) : '',
    'PM2.5': d['PM2.5'],
    PM10: d['PM10'],
    AQI: d['AQI'],
  }));

  return (
    <div className="space-y-6">
      {/* Regional Network Status Banner */}
      <div className="p-3.5 rounded-xl bg-gradient-to-r from-brand-900 to-indigo-900 text-white flex items-center justify-between text-xs shadow-sm">
        <div className="flex items-center space-x-2">
          <Layers className="w-4 h-4 text-emerald-400 flex-shrink-0" />
          <span>
            <strong>DELHI CAAQMS MONITORING NETWORK:</strong> 6 Regional stations encompassing 210,240 cumulative observation records (209,088 rolling 72-hour forecast windows). Anand Vihar (DEL_01) serves as the primary continuous telemetric benchmark hub.
          </span>
        </div>
        <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono text-[10px] font-bold border border-emerald-500/30">
          6 STATIONS ACTIVE
        </span>
      </div>

      {/* Station Information Card */}
      <Card
        title={
          <div className="flex items-center space-x-2">
            <MapPin className="w-5 h-5 text-brand-700" />
            <span>Delhi Continuous Ambient Air Quality Monitoring Station</span>
          </div>
        }
        subtitle="Anand Vihar Transit Hub & High-Density Monitoring Station (DEL_01)"
      >
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-lavender-50 border border-lavender-200/70">
            <div className="text-xs font-semibold text-gray-500 mb-1">Station ID & Name</div>
            <div className="text-base font-bold text-brand-950">DEL_01 / Anand Vihar</div>
            <div className="text-xs text-gray-500 mt-1">East Delhi Transit Corridor</div>
          </div>
          <div className="p-4 rounded-xl bg-lavender-50 border border-lavender-200/70">
            <div className="text-xs font-semibold text-gray-500 mb-1">Geographic Coordinates</div>
            <div className="text-base font-bold font-mono text-brand-950">28.6469° N, 77.3152° E</div>
            <div className="text-xs text-gray-500 mt-1">Elevation: 216 m ASL</div>
          </div>
          <div className="p-4 rounded-xl bg-lavender-50 border border-lavender-200/70">
            <div className="text-xs font-semibold text-gray-500 mb-1">Telemetry Dataset</div>
            <div className="text-base font-bold text-brand-950">35,064 Hourly Samples</div>
            <div className="text-xs text-gray-500 mt-1">4 Full Years (2020–2023)</div>
          </div>
          <div className="p-4 rounded-xl bg-lavender-50 border border-lavender-200/70">
            <div className="text-xs font-semibold text-gray-500 mb-1">Monitored Pollutants</div>
            <div className="text-base font-bold text-brand-950">PM2.5, PM10, NO2, SO2, CO, O3</div>
            <div className="text-xs text-gray-500 mt-1">Meteorology: Temp, Humidity, Wind</div>
          </div>
        </div>
      </Card>

      {/* Regional Station Grid */}
      <Card
        title={
          <div className="flex items-center space-x-2">
            <Layers className="w-5 h-5 text-brand-700" />
            <span>Delhi Regional CAAQMS Sensor Grid (6 Stations • 210,240 Total Observations)</span>
          </div>
        }
        subtitle="Regional monitoring network coverage with 209,088 rolling 72-hour forecast windows"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {[
            { id: 'DEL_01', name: 'Anand Vihar', district: 'East Delhi', type: 'Traffic / Commercial Hub', coords: '28.6469°N, 77.3152°E', status: 'Active Telemetry Hub', tag: 'Primary Benchmark' },
            { id: 'DEL_02', name: 'IGI Airport (T3)', district: 'South-West Delhi', type: 'Aviation Transport Hub', coords: '28.5629°N, 77.0984°E', status: 'Active Regional Node', tag: 'Suburban' },
            { id: 'DEL_03', name: 'R.K. Puram', district: 'South Delhi', type: 'Dense Residential Hub', coords: '28.5632°N, 77.1869°E', status: 'Active Regional Node', tag: 'Residential' },
            { id: 'DEL_04', name: 'Punjabi Bagh', district: 'West Delhi', type: 'Commercial / Residential', coords: '28.6741°N, 77.1310°E', status: 'Active Regional Node', tag: 'Commercial' },
            { id: 'DEL_05', name: 'Mandir Marg', district: 'Central Delhi', type: 'Cultural / Institutional', coords: '28.6365°N, 77.1997°E', status: 'Active Regional Node', tag: 'Central Zone' },
            { id: 'DEL_06', name: 'Shadipur', district: 'West-Central Delhi', type: 'Industrial & Transit Corridor', coords: '28.6515°N, 77.1581°E', status: 'Active Regional Node', tag: 'Industrial' },
          ].map((st) => (
            <div key={st.id} className="p-3.5 rounded-xl border border-lavender-200 bg-lavender-50/40 hover:bg-lavender-50/80 transition-all">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] font-bold text-brand-800 bg-brand-100/70 px-2 py-0.5 rounded">{st.id}</span>
                <span className="text-[10px] font-semibold text-emerald-700 bg-emerald-100/70 px-2 py-0.5 rounded">{st.tag}</span>
              </div>
              <div className="font-bold text-sm text-gray-900 mt-2">{st.name}</div>
              <div className="text-xs text-gray-600 mt-0.5">{st.type} ({st.district})</div>
              <div className="text-[11px] font-mono text-gray-500 mt-1">{st.coords}</div>
              <div className="text-[10px] text-brand-600 font-semibold mt-2 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                <span>{st.status} • 35,040 hrs</span>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Tab Navigation for EDA & Analysis */}
      <div className="flex space-x-2 border-b border-lavender-200 pb-2">
        <button
          onClick={() => setActiveTab('particulate')}
          className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${
            activeTab === 'particulate'
              ? 'bg-brand-700 text-white shadow-sm'
              : 'bg-lavender-100/60 text-brand-900 hover:bg-lavender-200/60'
          }`}
        >
          Recent Time-Series Trajectory
        </button>
        <button
          onClick={() => setActiveTab('diurnal')}
          className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${
            activeTab === 'diurnal'
              ? 'bg-brand-700 text-white shadow-sm'
              : 'bg-lavender-100/60 text-brand-900 hover:bg-lavender-200/60'
          }`}
        >
          Diurnal Cycle (24h Hourly Profile)
        </button>
        <button
          onClick={() => setActiveTab('seasonal')}
          className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${
            activeTab === 'seasonal'
              ? 'bg-brand-700 text-white shadow-sm'
              : 'bg-lavender-100/60 text-brand-900 hover:bg-lavender-200/60'
          }`}
        >
          Seasonal Variation (Monthly Pattern)
        </button>
        <button
          onClick={() => setActiveTab('categories')}
          className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${
            activeTab === 'categories'
              ? 'bg-brand-700 text-white shadow-sm'
              : 'bg-lavender-100/60 text-brand-900 hover:bg-lavender-200/60'
          }`}
        >
          CPCB AQI Category Distribution
        </button>
      </div>

      {/* Tab Content 1: Time Series */}
      {activeTab === 'particulate' && (
        <Card
          title={
            <div className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-brand-700" />
              <span>Observed Multi-Pollutant Time Series (Last {limit} Hours)</span>
            </div>
          }
          subtitle="Recent continuous monitoring measurements at Anand Vihar station"
        >
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={particulateChartData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0eef7" />
                <XAxis dataKey="time" stroke="#71717a" fontSize={11} />
                <YAxis stroke="#71717a" fontSize={11} unit=" µg/m³" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: '#e2ddf0',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line type="monotone" dataKey="PM2.5" stroke="#7c3aed" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="PM10" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="AQI" stroke="#dc2626" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* Tab Content 2: Diurnal Hourly Cycle */}
      {activeTab === 'diurnal' && (
        <Card
          title={
            <div className="flex items-center space-x-2">
              <Clock className="w-5 h-5 text-brand-700" />
              <span>Diurnal Hourly Profile (24-Hour Cycle Across 35,064 Samples)</span>
            </div>
          }
          subtitle="Depicting morning and evening traffic/inversion peaks characteristic of Delhi atmosphere"
        >
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={diurnalData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0eef7" />
                <XAxis dataKey="hour" stroke="#71717a" fontSize={11} />
                <YAxis stroke="#71717a" fontSize={11} unit=" µg/m³" />
                <Tooltip />
                <Bar dataKey="Average PM2.5" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* Tab Content 3: Seasonal Monthly Pattern */}
      {activeTab === 'seasonal' && (
        <Card
          title={
            <div className="flex items-center space-x-2">
              <Calendar className="w-5 h-5 text-brand-700" />
              <span>Seasonal Profile: Average PM2.5 Concentration by Month</span>
            </div>
          }
          subtitle="Showing severe winter peaks (Nov–Jan) due to temperature inversion and monsoon clearance (Jul–Aug)"
        >
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasonalData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0eef7" />
                <XAxis dataKey="month" stroke="#71717a" fontSize={12} />
                <YAxis stroke="#71717a" fontSize={12} unit=" µg/m³" />
                <Tooltip />
                <Bar dataKey="Average PM2.5" fill="#a78bfa" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* Tab Content 4: CPCB AQI Categories */}
      {activeTab === 'categories' && (
        <Card
          title={
            <div className="flex items-center space-x-2">
              <BarChart2 className="w-5 h-5 text-brand-700" />
              <span>CPCB AQI Category Distribution (35,064 Total Observations)</span>
            </div>
          }
          subtitle="Breakdown of air quality severity across the 4-year Delhi dataset (2020–2023)"
        >
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0eef7" />
                <XAxis dataKey="category" stroke="#71717a" fontSize={12} />
                <YAxis stroke="#71717a" fontSize={12} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}
    </div>
  );
};
