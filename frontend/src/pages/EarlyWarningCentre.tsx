import React, { useEffect, useState } from 'react';
import { getWarnings, apiService } from '../api/client';
import type { WarningListResponse, WarningThresholdsResponse } from '../types';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { getSeverityStyle } from '../utils/severity';
import {
  ShieldAlert,
  MapPin,
  Clock,
  CheckCircle2,
  AlertTriangle,
  BookOpen,
  Filter,
} from 'lucide-react';

const DEFAULT_WARNINGS: WarningListResponse = {
  station: "Anand Vihar, Delhi",
  total_warnings: 6,
  highest_severity: "Poor",
  warnings: [
    {
      station: "Anand Vihar, Delhi",
      lead_time_hours: 1,
      predicted_pm25: 98.4,
      warning_category: "Poor",
      alert_code: "TIER_4_POOR",
      severity_label: "Tier 4: Breathing Discomfort for Most People",
      color_hex: "#F97316",
      trend_type: "Steady",
      is_acute_spike: false,
      rationale: "Forecast PM2.5 of 98.4 µg/m³ exceeds the 90 µg/m³ Poor threshold.",
      recommended_action: "Avoid prolonged outdoor exposure; wear N95 mask in traffic corridors.",
      reference_datetime: "2023-12-31 23:00:00"
    },
    {
      station: "Anand Vihar, Delhi",
      lead_time_hours: 6,
      predicted_pm25: 86.2,
      warning_category: "Moderate",
      alert_code: "TIER_3_MODERATE",
      severity_label: "Tier 3: Moderate Discomfort",
      color_hex: "#F59E0B",
      trend_type: "Steady",
      is_acute_spike: false,
      rationale: "Forecast PM2.5 of 86.2 µg/m³ falls into Moderate tier.",
      recommended_action: "People with lung/heart disease should reduce outdoor exertion.",
      reference_datetime: "2023-12-31 23:00:00"
    },
    {
      station: "Anand Vihar, Delhi",
      lead_time_hours: 12,
      predicted_pm25: 78.5,
      warning_category: "Moderate",
      alert_code: "TIER_3_MODERATE",
      severity_label: "Tier 3: Moderate Discomfort",
      color_hex: "#F59E0B",
      trend_type: "Steady",
      is_acute_spike: false,
      rationale: "Forecast PM2.5 of 78.5 µg/m³ falls into Moderate tier.",
      recommended_action: "People with lung/heart disease should reduce outdoor exertion.",
      reference_datetime: "2023-12-31 23:00:00"
    },
    {
      station: "Anand Vihar, Delhi",
      lead_time_hours: 24,
      predicted_pm25: 88.0,
      warning_category: "Moderate",
      alert_code: "TIER_3_MODERATE",
      severity_label: "Tier 3: Moderate Discomfort",
      color_hex: "#F59E0B",
      trend_type: "Steady",
      is_acute_spike: false,
      rationale: "Forecast PM2.5 of 88.0 µg/m³ falls into Moderate tier.",
      recommended_action: "People with lung/heart disease should reduce outdoor exertion.",
      reference_datetime: "2023-12-31 23:00:00"
    },
    {
      station: "Anand Vihar, Delhi",
      lead_time_hours: 48,
      predicted_pm25: 95.3,
      warning_category: "Poor",
      alert_code: "TIER_4_POOR",
      severity_label: "Tier 4: Breathing Discomfort for Most People",
      color_hex: "#F97316",
      trend_type: "Steady",
      is_acute_spike: false,
      rationale: "Forecast PM2.5 of 95.3 µg/m³ exceeds 90 µg/m³ threshold.",
      recommended_action: "Mask advisory in effect for high-traffic and industrial areas.",
      reference_datetime: "2023-12-31 23:00:00"
    },
    {
      station: "Anand Vihar, Delhi",
      lead_time_hours: 72,
      predicted_pm25: 102.1,
      warning_category: "Poor",
      alert_code: "TIER_4_POOR",
      severity_label: "Tier 4: Breathing Discomfort for Most People",
      color_hex: "#F97316",
      trend_type: "Steady",
      is_acute_spike: false,
      rationale: "Forecast PM2.5 of 102.1 µg/m³ exceeds 90 µg/m³ threshold.",
      recommended_action: "Limit prolonged heavy exertion; sensitive groups remain indoors.",
      reference_datetime: "2023-12-31 23:00:00"
    }
  ]
};

export const EarlyWarningCentre: React.FC = () => {
  const [offset, setOffset] = useState<number>(0);
  const [warningsData, setWarningsData] = useState<WarningListResponse | null>(DEFAULT_WARNINGS);
  const [thresholds, setThresholds] = useState<WarningThresholdsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [filterSeverity, setFilterSeverity] = useState<string>('all');

  useEffect(() => {
    let isCancelled = false;
    const fetchThresholdsAndWarnings = async () => {
      try {
        setLoading(true);
        const [thRes, wRes] = await Promise.all([
          apiService.getWarningThresholds().catch(() => null),
          getWarnings().catch(() => null),
        ]);

        if (isCancelled) return;
        if (thRes) setThresholds(thRes);
        if (wRes?.data) setWarningsData(wRes.data);
      } catch (err) {
        console.error('Failed to load warning data', err);
      } finally {
        if (!isCancelled) setLoading(false);
      }
    };

    fetchThresholdsAndWarnings();
    return () => {
      isCancelled = true;
    };
  }, [offset]);

  const allWarnings = warningsData?.warnings || [];
  const filteredWarnings =
    filterSeverity === 'all'
      ? allWarnings
      : allWarnings.filter(
          (w) => w.warning_category.toLowerCase() === filterSeverity.toLowerCase()
        );

  return (
    <div className="space-y-6">
      {/* Regulatory Alert Banner */}
      <div className="p-3.5 rounded-xl bg-gradient-to-r from-brand-900 to-indigo-900 text-white flex items-center justify-between text-xs shadow-sm">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 text-rose-400 flex-shrink-0" />
          <span>
            <strong>CPCB NAAQS REGULATORY EARLY WARNING SYSTEM:</strong> Continuous multi-horizon automated hazard alerts and graded response action protocol (GRAP) for Delhi NCR.
          </span>
        </div>
        <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-mono text-[10px] font-bold border border-rose-500/30">
          DELHI GRAP ACTIVE
        </span>
      </div>

      {/* Control & Station Bar */}
      <Card
        title={
          <div className="flex items-center space-x-2">
            <ShieldAlert className="w-5 h-5 text-brand-700" />
            <span>Delhi Regulatory Early Warning Control Deck</span>
          </div>
        }
        subtitle="CPCB National Ambient Air Quality Standards & Graded Response Action Plan (GRAP)"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <div className="space-y-1.5">
            <label className="text-xs font-bold text-gray-700 flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-brand-600" />
              Target Location
            </label>
            <div className="w-full h-9 bg-lavender-100/60 border border-lavender-300 text-xs font-semibold rounded-lg px-3 py-2 text-brand-950 flex items-center justify-between">
              <span>Anand Vihar, Delhi (DEL_01)</span>
              <span className="text-[10px] bg-brand-200/80 text-brand-900 px-1.5 py-0.5 rounded font-mono">CPCB Hub</span>
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-bold text-gray-700 flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-brand-600" />
              Severity Tier Filter
            </label>
            <select
              value={filterSeverity}
              onChange={(e) => setFilterSeverity(e.target.value)}
              className="w-full h-9 bg-lavender-50/60 border border-lavender-200 text-xs font-semibold rounded-lg px-3 py-1.5 text-brand-900 focus:outline-none focus:ring-2 focus:ring-brand-500 cursor-pointer"
            >
              <option value="all">All Severities</option>
              <option value="Severe">Severe (GRAP Stage IV)</option>
              <option value="Very Poor">Very Poor (GRAP Stage III)</option>
              <option value="Poor">Poor</option>
              <option value="Moderate">Moderate</option>
              <option value="Satisfactory">Satisfactory</option>
              <option value="Good">Good</option>
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-bold text-gray-700 flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-brand-600" />
              Historical Replay Step
            </label>
            <input
              type="number"
              min="0"
              value={offset}
              onChange={(e) => setOffset(Math.max(0, parseInt(e.target.value) || 0))}
              className="w-full h-9 bg-lavender-50/60 border border-lavender-200 text-xs font-mono rounded-lg px-3 py-1.5 text-gray-900 focus:outline-none focus:ring-2 focus:ring-brand-500"
              placeholder="0 (latest)"
            />
          </div>
        </div>
      </Card>

      {/* Warnings Grid */}
      <div className="space-y-4">
        <h3 className="text-sm font-bold text-gray-800 flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-brand-700" />
          <span>Active Forecast Alerts ({filteredWarnings.length} Horizons Triggered)</span>
        </h3>

        {loading ? (
          <div className="h-40 flex items-center justify-center text-xs text-gray-400 bg-white rounded-xl border border-lavender-200">
            Evaluating regulatory warning thresholds...
          </div>
        ) : filteredWarnings.length === 0 ? (
          <div className="p-8 text-center text-xs text-gray-500 bg-white rounded-xl border border-lavender-200">
            No warnings matching selected severity filter.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredWarnings.map((w) => {
              const sev = getSeverityStyle(w.warning_category, w.color_hex);
              return (
                <div
                  key={w.lead_time_hours}
                  className="p-5 rounded-2xl border transition-all shadow-sm flex flex-col justify-between"
                  style={{
                    borderColor: w.color_hex ? `${w.color_hex}40` : '#e2ddf0',
                    backgroundColor: w.color_hex ? `${w.color_hex}06` : '#ffffff',
                  }}
                >
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-brand-100 text-brand-900">
                        +{w.lead_time_hours}h Horizon
                      </span>
                      <span
                        className="text-xs font-bold px-2.5 py-0.5 rounded-md"
                        style={{
                          backgroundColor: w.color_hex ? `${w.color_hex}25` : '#eee',
                          color: w.color_hex || '#333',
                        }}
                      >
                        {w.warning_category}
                      </span>
                    </div>

                    <div className="mb-3">
                      <div className="text-xs text-gray-500 font-semibold">Forecast PM2.5</div>
                      <div className="text-2xl font-black text-gray-900">
                        {w.predicted_pm25.toFixed(1)} <span className="text-xs font-normal text-gray-500">µg/m³</span>
                      </div>
                    </div>

                    <div className="text-xs text-gray-700 leading-relaxed mb-3">
                      <strong>Rationale:</strong> {w.rationale}
                    </div>
                  </div>

                  <div className="pt-3 border-t border-gray-100 text-xs font-medium text-brand-950">
                    <span className="font-bold text-gray-600 block mb-0.5">Mandated Protocol:</span>
                    {w.recommended_action}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* CPCB Regulatory Tier Reference Card */}
      <Card
        title={
          <div className="flex items-center space-x-2">
            <BookOpen className="w-5 h-5 text-brand-700" />
            <span>Central Pollution Control Board (CPCB) Regulatory Framework</span>
          </div>
        }
        subtitle="National Ambient Air Quality Standards (NAAQS) 24-Hour PM2.5 Threshold Reference"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-lavender-50/70 border-b border-lavender-200 text-brand-900 uppercase font-semibold">
              <tr>
                <th className="py-2 px-3">Tier</th>
                <th className="py-2 px-3">Category</th>
                <th className="py-2 px-3">PM2.5 Range (µg/m³)</th>
                <th className="py-2 px-3">Health Impact & Mandatory Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-lavender-100">
              <tr>
                <td className="py-2 px-3 font-bold text-emerald-700">Tier 1</td>
                <td className="py-2 px-3"><span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-semibold text-[11px]">Good</span></td>
                <td className="py-2 px-3 font-mono">0 – 30</td>
                <td className="py-2 px-3 text-gray-600">Minimal health impact; normal outdoor recreation permitted.</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-bold text-lime-700">Tier 2</td>
                <td className="py-2 px-3"><span className="px-2 py-0.5 rounded bg-lime-100 text-lime-800 font-semibold text-[11px]">Satisfactory</span></td>
                <td className="py-2 px-3 font-mono">30 – 60</td>
                <td className="py-2 px-3 text-gray-600">Minor breathing discomfort to sensitive individuals.</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-bold text-amber-700">Tier 3</td>
                <td className="py-2 px-3"><span className="px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-semibold text-[11px]">Moderate</span></td>
                <td className="py-2 px-3 font-mono">60 – 90</td>
                <td className="py-2 px-3 text-gray-600">Breathing discomfort to people with lungs, asthma and heart diseases.</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-bold text-orange-700">Tier 4</td>
                <td className="py-2 px-3"><span className="px-2 py-0.5 rounded bg-orange-100 text-orange-800 font-semibold text-[11px]">Poor</span></td>
                <td className="py-2 px-3 font-mono">90 – 120</td>
                <td className="py-2 px-3 text-gray-600">Breathing discomfort to most people on prolonged exposure. Mask advisory.</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-bold text-rose-700">Tier 5</td>
                <td className="py-2 px-3"><span className="px-2 py-0.5 rounded bg-rose-100 text-rose-800 font-semibold text-[11px]">Very Poor</span></td>
                <td className="py-2 px-3 font-mono">120 – 250</td>
                <td className="py-2 px-3 text-gray-600">GRAP Stage III: Cease construction dust activities, restrict non-compliant vehicles.</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-bold text-red-900">Tier 6</td>
                <td className="py-2 px-3"><span className="px-2 py-0.5 rounded bg-red-200 text-red-950 font-semibold text-[11px]">Severe</span></td>
                <td className="py-2 px-3 font-mono">250+</td>
                <td className="py-2 px-3 text-gray-600">GRAP Stage IV: Severe emergency protocols, ban diesel trucks, mandate remote work.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};
