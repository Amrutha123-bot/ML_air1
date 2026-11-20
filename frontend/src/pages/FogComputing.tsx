import React, { useState } from 'react';
import { Activity, Cpu, Filter, Radio, Network, Bell, ArrowRight } from 'lucide-react';

export const FogComputing: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('pipeline');

  const pipelineStages = [
    { title: 'Air Quality Sensors', icon: Activity, desc: 'IoT ambient telemetries' },
    { title: 'Fog Nodes', icon: Cpu, desc: 'Local edge preprocessing' },
    { title: 'Data Screening', icon: Filter, desc: 'Spike detection & smoothing' },
    { title: 'Gateway', icon: Radio, desc: 'Secure payload compression' },
    { title: 'Central ML Processing', icon: Network, desc: 'Spatiotemporal Graph Neural Net' },
    { title: 'Forecasting & Warnings', icon: Bell, desc: 'Public health threshold alerts' },
  ];

  return (
    <div className="space-y-6">
      {/* Top Tabs */}
      <div className="flex border-b border-[#1E314B] space-x-6">
        {[
          { id: 'pipeline', label: 'Processing Pipeline' },
          { id: 'quality', label: 'Data Quality' },
          { id: 'gateway', label: 'Fog Gateway' },
          { id: 'transmission', label: 'Transmission Analysis' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`pb-3 text-sm font-semibold transition-all relative ${
              activeTab === tab.id ? 'text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.label}
            {activeTab === tab.id && (
              <span className="absolute bottom-0 left-0 w-full h-0.5 bg-[#14D6C5] rounded-full" />
            )}
          </button>
        ))}
      </div>

      {/* Horizontal Connected Pipeline Card */}
      <div className="bg-[#142238] rounded-xl border border-[#1E314B] p-6 shadow-sm">
        <h3 className="text-sm font-bold text-white mb-6">Edge-to-Cloud Distributed Architecture</h3>

        <div className="flex flex-col md:flex-row items-center justify-between gap-3">
          {pipelineStages.map((stage, idx) => {
            const Icon = stage.icon;
            return (
              <React.Fragment key={idx}>
                <div className="flex-1 w-full bg-[#0B1424] border border-[#1E314B] rounded-xl p-4 flex flex-col items-center text-center shadow-xs">
                  <div className="w-10 h-10 rounded-lg bg-teal-accent/15 border border-teal-accent/30 text-teal-accent flex items-center justify-center mb-3">
                    <Icon className="w-5 h-5" />
                  </div>
                  <h4 className="text-xs font-bold text-white mb-1">{stage.title}</h4>
                  <p className="text-[11px] text-slate-400">{stage.desc}</p>
                </div>
                {idx < pipelineStages.length - 1 && (
                  <ArrowRight className="hidden md:block w-4 h-4 text-slate-500 flex-shrink-0" />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );
};
