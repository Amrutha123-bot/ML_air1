import React, { useEffect, useState } from 'react';
import { getMetrics, getLeaderboard, getConfusionMatrix, getFractionResults } from '../api/client';
import { Card } from '../components/ui/Card';
import {
  DELHI_LEADERBOARD,
  DELHI_METRICS_DATA,
  DELHI_CONFUSION_DATA,
  DELHI_FRACTIONS_DATA,
} from '../data/delhiRealData';
import {
  Cpu,
  Award,
  BarChart2,
  AlertTriangle,
  TrendingDown,
  Layers,
  CheckCircle,
  Table,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
} from 'recharts';

export const ModelLaboratory: React.FC = () => {
  const [leaderboardData, setLeaderboardData] = useState<any[]>(DELHI_LEADERBOARD);
  const [metricsData, setMetricsData] = useState<any[]>(DELHI_METRICS_DATA);
  const [confusionData, setConfusionData] = useState<any>(DELHI_CONFUSION_DATA);
  const [fractionsData, setFractionsData] = useState<any[]>(
    DELHI_FRACTIONS_DATA.map((e: any) => ({
      fraction: `${e.fraction_pct ?? 100}%`,
      train_samples: e.training_samples ?? e.train_samples ?? 0,
      MAE: Number((e.mean_test_mae ?? e.h24_mae ?? 0).toFixed(2)),
      RMSE: Number((e.mean_test_rmse ?? e.h24_rmse ?? 0).toFixed(2)),
      R2: Number((e.mean_test_r2 ?? e.h24_r2 ?? 0).toFixed(4)),
    }))
  );
  const [selectedHorizon, setSelectedHorizon] = useState<string>('+24h');
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    let isCancelled = false;
    const loadAllMetrics = async () => {
      try {
        const [lbRes, metRes, cmRes, frRes] = await Promise.all([
          getLeaderboard().catch(() => null),
          getMetrics().catch(() => null),
          getConfusionMatrix().catch(() => null),
          getFractionResults().catch(() => null),
        ]);

        if (isCancelled) return;

        if (lbRes?.data?.leaderboard && Array.isArray(lbRes.data.leaderboard)) {
          setLeaderboardData(lbRes.data.leaderboard);
        }
        if (metRes?.data?.metrics && Array.isArray(metRes.data.metrics)) {
          setMetricsData(metRes.data.metrics);
        }
        if (cmRes?.data?.matrix) {
          setConfusionData(cmRes.data);
        }
        const experiments = frRes?.data?.experiments ?? (Array.isArray(frRes?.data) ? frRes?.data : null);
        if (experiments && Array.isArray(experiments)) {
          const frList = experiments.map((e: any) => ({
            fraction: `${e.fraction_pct ?? (e.fraction ? Math.round(e.fraction * 100) : 100)}%`,
            train_samples: e.training_samples ?? e.train_samples ?? 0,
            MAE: Number((e.mean_test_mae ?? e.h24_mae ?? 0).toFixed(2)),
            RMSE: Number((e.mean_test_rmse ?? e.h24_rmse ?? 0).toFixed(2)),
            R2: Number((e.mean_test_r2 ?? e.h24_r2 ?? 0).toFixed(4)),
          }));
          setFractionsData(frList);
        }
      } catch (err) {
        console.warn('Backend unavailable, using cached benchmark metrics', err);
      }
    };

    loadAllMetrics();
    return () => {
      isCancelled = true;
    };
  }, []);

  // Filter metrics by selected horizon for the bar chart
  const currentHorizonMetrics = metricsData
    .filter((m) => m.horizon === selectedHorizon)
    .sort((a, b) => a.mae - b.mae);

  // Prepare chart data for lead time MAE across horizons for top models
  const horizonsList = ['+1h', '+6h', '+12h', '+24h', '+48h', '+72h'];
  const leadTimeChartData = horizonsList.map((h) => {
    const rf = metricsData.find((m) => m.model.includes('Random Forest') && m.horizon === h);
    const xgb = metricsData.find((m) => m.model === 'XGBoost' && m.horizon === h);
    const pers = metricsData.find((m) => m.model.includes('Persistence') && !m.model.includes('Seasonal') && m.horizon === h);
    const lstm = metricsData.find((m) => m.model.includes('LSTM') && !m.model.includes('CNN') && m.horizon === h);
    return {
      horizon: h,
      'Random Forest': rf ? Number(rf.mae.toFixed(1)) : null,
      'XGBoost': xgb ? Number(xgb.mae.toFixed(1)) : null,
      'Persistence': pers ? Number(pers.mae.toFixed(1)) : null,
      'LSTM': lstm ? Number(lstm.mae.toFixed(1)) : null,
    };
  });

  return (
    <div className="space-y-6">
      {/* Empirical Benchmark Header Banner */}
      <div className="p-3.5 rounded-xl bg-gradient-to-r from-brand-900 to-indigo-900 text-white flex items-center justify-between text-xs shadow-sm">
        <div className="flex items-center space-x-2">
          <Award className="w-4 h-4 text-amber-400 flex-shrink-0" />
          <span>
            <strong>EMPIRICAL MODEL EVALUATION LABORATORY:</strong> All metrics, error distributions, confusion matrices, and model comparison leaderboards are computed from machine learning test runs on the Delhi benchmark dataset.
          </span>
        </div>
        <span className="px-2 py-0.5 rounded bg-brand-500/30 text-brand-200 font-mono text-[10px] font-bold border border-brand-400/30">
          DELHI BENCHMARK
        </span>
      </div>

      {/* Champion Model Performance Banner */}
      <div className="p-5 rounded-2xl bg-gradient-to-r from-brand-900 via-brand-800 to-indigo-900 text-white shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Award className="w-5 h-5 text-amber-400" />
              <span className="text-xs uppercase tracking-wider font-bold text-amber-400">
                Empirical Champion Model (+24h Horizon)
              </span>
            </div>
            <h2 className="text-2xl font-black tracking-tight">Random Forest Ensemble (100 Trees)</h2>
            <p className="text-xs text-brand-200 mt-1">
              Outperformed deep neural architectures and classical baselines with MAE: <strong>14.60 µg/m³</strong>, R²: <strong>0.9544</strong>, and <strong>+23.7% skill score</strong> over persistence.
            </p>
          </div>
          <div className="flex gap-4">
            <div className="bg-white/10 px-4 py-2 rounded-xl backdrop-blur-sm border border-white/10">
              <div className="text-[10px] text-brand-200 uppercase font-semibold">Test MAE</div>
              <div className="text-xl font-bold font-mono text-white">14.60</div>
            </div>
            <div className="bg-white/10 px-4 py-2 rounded-xl backdrop-blur-sm border border-white/10">
              <div className="text-[10px] text-brand-200 uppercase font-semibold">R² Score</div>
              <div className="text-xl font-bold font-mono text-emerald-400">0.9544</div>
            </div>
            <div className="bg-white/10 px-4 py-2 rounded-xl backdrop-blur-sm border border-white/10">
              <div className="text-[10px] text-brand-200 uppercase font-semibold">Classification Acc</div>
              <div className="text-xl font-bold font-mono text-white">74.1%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Comparison Table */}
      <Card
        title={
          <div className="flex items-center justify-between w-full">
            <div className="flex items-center space-x-2">
              <Table className="w-5 h-5 text-brand-700" />
              <span>Delhi Model Benchmark Leaderboard ({selectedHorizon} Horizon)</span>
            </div>
            <div className="flex space-x-1">
              {horizonsList.map((h) => (
                <button
                  key={h}
                  onClick={() => setSelectedHorizon(h)}
                  className={`px-2.5 py-1 rounded text-xs font-bold transition-all ${
                    selectedHorizon === h
                      ? 'bg-brand-700 text-white'
                      : 'bg-lavender-100 text-brand-900 hover:bg-lavender-200'
                  }`}
                >
                  {h}
                </button>
              ))}
            </div>
          </div>
        }
        subtitle="Chronological test partition evaluation (5,094 samples with 72h purge gap)"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-lavender-50/70 border-b border-lavender-200 text-brand-900 uppercase font-semibold">
              <tr>
                <th className="py-2.5 px-3">Rank</th>
                <th className="py-2.5 px-3">Model</th>
                <th className="py-2.5 px-3">Test MAE</th>
                <th className="py-2.5 px-3">Test RMSE</th>
                <th className="py-2.5 px-3">R² Score</th>
                <th className="py-2.5 px-3">sMAPE (%)</th>
                <th className="py-2.5 px-3">Skill vs Baseline</th>
                <th className="py-2.5 px-3">Training Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-lavender-100 font-mono">
              {currentHorizonMetrics.map((m, idx) => {
                const isChampion = m.model.includes('Random Forest') || m.model === 'XGBoost';
                return (
                  <tr key={`${m.model}-${m.horizon}`} className={isChampion ? 'bg-emerald-50/40 font-bold' : ''}>
                    <td className="py-2.5 px-3 font-bold text-gray-900">#{idx + 1}</td>
                    <td className="py-2.5 px-3 font-sans font-semibold text-brand-950 flex items-center gap-1.5">
                      {isChampion && <Award className="w-3.5 h-3.5 text-amber-500" />}
                      {m.model}
                    </td>
                    <td className="py-2.5 px-3 font-bold text-brand-800">{m.mae.toFixed(2)}</td>
                    <td className="py-2.5 px-3 text-gray-600">{m.rmse.toFixed(2)}</td>
                    <td className={`py-2.5 px-3 font-bold ${m.r2 > 0.9 ? 'text-emerald-700' : 'text-gray-700'}`}>
                      {m.r2.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-gray-600">{m.smape.toFixed(1)}%</td>
                    <td className={`py-2.5 px-3 font-bold ${m.skill_vs_persistence > 0 ? 'text-emerald-700' : 'text-rose-600'}`}>
                      {m.skill_vs_persistence > 0 ? `+${(m.skill_vs_persistence * 100).toFixed(1)}%` : `${(m.skill_vs_persistence * 100).toFixed(1)}%`}
                    </td>
                    <td className="py-2.5 px-3 text-gray-500 font-sans">{m.training_time_s.toFixed(1)}s</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Multi-Horizon Degradation Chart */}
      <Card
        title={
          <div className="flex items-center space-x-2">
            <BarChart2 className="w-5 h-5 text-brand-700" />
            <span>Multi-Horizon MAE Degradation Curve (1h to 72h Ahead)</span>
          </div>
        }
        subtitle="Comparing forecast error growth across prediction lead times"
      >
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={leadTimeChartData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0eef7" />
              <XAxis dataKey="horizon" stroke="#71717a" fontSize={12} />
              <YAxis stroke="#71717a" fontSize={12} unit=" µg/m³" />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="Random Forest" stroke="#10b981" strokeWidth={3} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="XGBoost" stroke="#6366f1" strokeWidth={2} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="Persistence" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 4 }} />
              <Line type="monotone" dataKey="LSTM" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* CPCB 6-Tier Confusion Matrix & Data Efficiency */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Confusion Matrix Heatmap */}
        <Card
          title={
            <div className="flex items-center space-x-2">
              <Layers className="w-5 h-5 text-brand-700" />
              <span>CPCB 6-Tier Confusion Matrix (+24h Horizon)</span>
            </div>
          }
          subtitle={`Evaluated on Champion Model | Accuracy: ${(confusionData?.classification_metrics?.overall_accuracy * 100 || 74.1).toFixed(1)}% | Macro-F1: ${(confusionData?.classification_metrics?.macro_f1 || 0.628).toFixed(3)}`}
        >
          {confusionData?.matrix ? (
            <div className="space-y-4">
              <div className="overflow-x-auto">
                <table className="w-full text-center text-xs">
                  <thead>
                    <tr className="border-b border-lavender-200">
                      <th className="py-2 text-left font-bold text-gray-500">True \ Pred</th>
                      {confusionData.labels.map((lbl: string) => (
                        <th key={lbl} className="py-2 px-1 font-bold text-gray-700 text-[10px]">
                          {lbl.slice(0, 4)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {confusionData.matrix.map((row: number[], rIdx: number) => {
                      const trueLabel = confusionData.labels[rIdx];
                      return (
                        <tr key={trueLabel} className="border-b border-lavender-50">
                          <td className="py-2 text-left font-bold text-gray-700 text-[11px]">{trueLabel}</td>
                          {row.map((val: number, cIdx: number) => {
                            const isDiag = rIdx === cIdx;
                            const intensity = Math.min(1, val);
                            const bgColor = isDiag
                              ? `rgba(16, 185, 129, ${Math.max(0.15, intensity)})`
                              : `rgba(239, 68, 68, ${Math.min(0.4, val * 0.6)})`;
                            return (
                              <td
                                key={cIdx}
                                className="py-2 px-1 font-mono text-[10px] font-bold"
                                style={{ backgroundColor: bgColor }}
                              >
                                {(val * 100).toFixed(0)}%
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="text-[11px] text-gray-500 flex justify-between">
                <span>Green diagonal = Correct Tier Predictions</span>
                <span>Weighted F1: {(confusionData?.classification_metrics?.weighted_f1 || 0.727).toFixed(3)}</span>
              </div>
            </div>
          ) : (
            <div className="h-40 flex items-center justify-center text-xs text-gray-400">
              Loading confusion matrix...
            </div>
          )}
        </Card>

        {/* Training Data Fractions */}
        <Card
          title={
            <div className="flex items-center space-x-2">
              <Cpu className="w-5 h-5 text-brand-700" />
              <span>Data Fraction Efficiency (15% to 100%)</span>
            </div>
          }
          subtitle="Model test error trajectory as training dataset volume scales"
        >
          {fractionsData.length > 0 ? (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={fractionsData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0eef7" />
                  <XAxis dataKey="fraction" stroke="#71717a" fontSize={11} />
                  <YAxis stroke="#71717a" fontSize={11} unit=" µg/m³" />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="MAE" stroke="#7c3aed" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="RMSE" stroke="#ea580c" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-40 flex items-center justify-center text-xs text-gray-400">
              Loading data fraction experiments...
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};
