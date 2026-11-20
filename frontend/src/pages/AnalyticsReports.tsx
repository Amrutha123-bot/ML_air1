import React, { useEffect, useState } from 'react';
import { Card } from '../components/ui/Card';
import { getEDA } from '../api/client';
import { DELHI_EDA_DATA } from '../data/delhiRealData';
import axios from 'axios';
import {
  BarChart3,
  Layers,
  Printer,
  Calendar,
  MapPin,
  Activity,
  AlertTriangle,
  Database,
  FileText,
  CheckCircle,
} from 'lucide-react';

export const AnalyticsReports: React.FC = () => {
  const [edaData, setEdaData] = useState<any>(DELHI_EDA_DATA);
  const [correlations, setCorrelations] = useState<any[]>(() => {
    const matrix = DELHI_EDA_DATA.correlation_matrix || {};
    return Object.entries(matrix).map(([pol, others]: [string, any]) => ({
      pollutant: pol,
      ...others,
    }));
  });
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    let isCancelled = false;
    const fetchAnalytics = async () => {
      try {
        const [edaRes, corrRes] = await Promise.all([
          getEDA().catch(() => null),
          axios.get('http://localhost:8000/api/v1/analytics/correlation').catch(() => null),
        ]);

        if (isCancelled) return;
        if (edaRes?.data) setEdaData(edaRes.data);
        if (corrRes?.data && Array.isArray(corrRes.data)) setCorrelations(corrRes.data);
      } catch (err) {
        console.warn('Backend unavailable, using cached real dataset metrics', err);
      }
    };

    fetchAnalytics();
    return () => {
      isCancelled = true;
    };
  }, []);

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="space-y-6">
      {/* Synthetic Data Disclaimer Banner */}
      <div className="p-4 rounded-xl bg-amber-50 border border-amber-200/80 text-amber-900 flex items-center justify-between text-xs">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0" />
          <span>
            <strong>SYNTHETIC DATASET DISCLOSURE:</strong> All metrics, tables, and reports presented below are derived from <strong>delhi_synthetic_air_quality.csv</strong> (35,064 records), a programmatically generated, AR(1)-calibrated dataset modeled on CPCB NAAQS guidelines for Anand Vihar, Delhi.
          </span>
        </div>
        <button
          onClick={handlePrint}
          className="px-3 py-1.5 rounded-lg bg-amber-200 text-amber-950 hover:bg-amber-300 font-semibold flex items-center gap-1.5 cursor-pointer"
        >
          <Printer className="w-3.5 h-3.5" />
          Print Report
        </button>
      </div>

      {/* Dataset Inventory & Quality Summary */}
      <Card
        title={
          <div className="flex items-center space-x-2">
            <Database className="w-5 h-5 text-brand-700" />
            <span>Delhi Synthetic Air Quality Dataset Inventory</span>
          </div>
        }
        subtitle="Complete specification of the underlying 35,064-sample machine learning corpus"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="p-4 rounded-xl bg-lavender-50 border border-lavender-200/80">
            <div className="text-xs text-gray-500 font-semibold">Total Observations</div>
            <div className="text-2xl font-black text-brand-950 font-mono mt-1">35,064</div>
            <div className="text-[11px] text-gray-500 mt-1">Hourly continuous series</div>
          </div>
          <div className="p-4 rounded-xl bg-lavender-50 border border-lavender-200/80">
            <div className="text-xs text-gray-500 font-semibold">Temporal Span</div>
            <div className="text-base font-bold text-brand-950 font-mono mt-1">2020-01-01 to 2023-12-31</div>
            <div className="text-[11px] text-gray-500 mt-1">4 Full Calendar Years</div>
          </div>
          <div className="p-4 rounded-xl bg-lavender-50 border border-lavender-200/80">
            <div className="text-xs text-gray-500 font-semibold">Geographic Scope</div>
            <div className="text-base font-bold text-brand-950 mt-1">Delhi NCR Only</div>
            <div className="text-[11px] text-gray-500 mt-1">Station: Anand Vihar (DEL_01)</div>
          </div>
          <div className="p-4 rounded-xl bg-lavender-50 border border-lavender-200/80">
            <div className="text-xs text-gray-500 font-semibold">Data Integrity</div>
            <div className="text-base font-bold text-emerald-700 mt-1">100% Clean</div>
            <div className="text-[11px] text-gray-500 mt-1">0 missing, 0 duplicates</div>
          </div>
        </div>

        {/* Feature Engineering Details */}
        <div className="border-t border-lavender-200 pt-4">
          <h4 className="text-xs font-bold text-brand-900 uppercase tracking-wider mb-3">
            Engineered Feature Pipeline (64 Features)
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
              <span className="font-bold text-gray-900 block mb-1">Temporal Autoregressive Lags</span>
              <p className="text-gray-600 leading-relaxed">
                1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h lags for PM2.5, PM10, NO2, Temperature, and Wind Speed to capture short and multi-day temporal persistence.
              </p>
            </div>
            <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
              <span className="font-bold text-gray-900 block mb-1">Rolling Windows & Dispersion</span>
              <p className="text-gray-600 leading-relaxed">
                6h, 24h, and 72h rolling means and standard deviations computed with shift(1) to strictly eliminate lookahead contamination.
              </p>
            </div>
            <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
              <span className="font-bold text-gray-900 block mb-1">Cyclic Embeddings & Interactions</span>
              <p className="text-gray-600 leading-relaxed">
                Sin/cos transforms of hour-of-day, day-of-year, and day-of-week, plus PM2.5/PM10 ratio, NO2/NOx ratio, and temperature-wind interaction.
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* Pollutant Summary Statistics */}
      <Card
        title={
          <div className="flex items-center space-x-2">
            <FileText className="w-5 h-5 text-brand-700" />
            <span>Pollutant Descriptive Statistics (35,064 Observations)</span>
          </div>
        }
        subtitle="Empirical distributions across criteria pollutants modeled for Delhi"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-lavender-50/70 border-b border-lavender-200 text-brand-900 uppercase font-sans font-semibold">
              <tr>
                <th className="py-2.5 px-3">Parameter</th>
                <th className="py-2.5 px-3">Unit</th>
                <th className="py-2.5 px-3">Mean</th>
                <th className="py-2.5 px-3">Std Dev</th>
                <th className="py-2.5 px-3">Min</th>
                <th className="py-2.5 px-3">P25</th>
                <th className="py-2.5 px-3">Median</th>
                <th className="py-2.5 px-3">P75</th>
                <th className="py-2.5 px-3">Max</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-lavender-100">
              {(() => {
                const summary = edaData?.variables_summary || edaData?.pollutant_statistics || DELHI_EDA_DATA.variables_summary;
                return summary ? (
                  Object.entries(summary).map(([pol, s]: [string, any]) => (
                    <tr key={pol}>
                      <td className="py-2 px-3 font-sans font-bold text-brand-950">{pol}</td>
                      <td className="py-2 px-3 font-sans text-gray-500">{pol.includes('CO') ? 'mg/m³' : (pol.includes('Temp') ? '°C' : (pol.includes('Wind') ? 'm/s' : (pol.includes('Hum') ? '%' : 'µg/m³')))}</td>
                      <td className="py-2 px-3 text-brand-800 font-bold">{Number(s.mean).toFixed(1)}</td>
                      <td className="py-2 px-3 text-gray-600">{Number(s.std).toFixed(1)}</td>
                      <td className="py-2 px-3 text-gray-600">{Number(s.min).toFixed(1)}</td>
                      <td className="py-2 px-3 text-gray-600">{Number(s['25%'] ?? s.p25 ?? 0).toFixed(1)}</td>
                      <td className="py-2 px-3 text-gray-700 font-semibold">{Number(s['50%'] ?? s.median ?? 0).toFixed(1)}</td>
                      <td className="py-2 px-3 text-gray-600">{Number(s['75%'] ?? s.p75 ?? 0).toFixed(1)}</td>
                      <td className="py-2 px-3 text-brand-950 font-bold">{Number(s.max).toFixed(1)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={9} className="py-4 text-center text-gray-400">Loading pollutant metrics...</td>
                  </tr>
                );
              })()}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Correlation Heatmap */}
      {correlations.length > 0 && (
        <Card
          title={
            <div className="flex items-center space-x-2">
              <Layers className="w-5 h-5 text-brand-700" />
              <span>Inter-Pollutant Pearson Correlation Matrix</span>
            </div>
          }
          subtitle="Showing expected physical coupling between PM2.5, PM10, NOx, and weather parameters"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-center text-xs font-mono">
              <thead>
                <tr className="border-b border-lavender-200">
                  <th className="py-2 text-left font-bold text-gray-500 font-sans">Pollutant</th>
                  {correlations.map((c) => (
                    <th key={c.pollutant} className="py-2 px-1 font-bold text-gray-700 text-[11px] font-sans">
                      {c.pollutant}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-lavender-50">
                {correlations.map((row) => (
                  <tr key={row.pollutant}>
                    <td className="py-2 text-left font-bold text-brand-950 font-sans">{row.pollutant}</td>
                    {correlations.map((other) => {
                      const val = Number(row[other.pollutant] ?? 0);
                      const isSelf = row.pollutant === other.pollutant;
                      const isHighPos = val > 0.6;
                      const isNeg = val < -0.3;
                      const bg = isSelf
                        ? '#f3f4f6'
                        : isHighPos
                        ? '#dcfce7'
                        : isNeg
                        ? '#fee2e2'
                        : '#ffffff';
                      return (
                        <td
                          key={other.pollutant}
                          className="py-2 px-1 text-[11px] font-bold"
                          style={{ backgroundColor: bg }}
                        >
                          {val.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
};
