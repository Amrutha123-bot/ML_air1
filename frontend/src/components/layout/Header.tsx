import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { RefreshCw, Bell } from 'lucide-react';

interface HeaderProps {
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

const pageTitles: Record<string, { title: string; subtitle: string }> = {
  '/': {
    title: 'Overview',
    subtitle: 'Real-time air quality summary and system status',
  },
  '/stations': {
    title: 'Monitoring Stations',
    subtitle: 'Explore monitoring stations, real-time readings and historical data',
  },
  '/forecasting': {
    title: 'AI Forecasting',
    subtitle: 'Air quality prediction using trained ML models',
  },
  '/models': {
    title: 'Model Laboratory',
    subtitle: 'Model information, training results and performance evaluation',
  },
  '/warnings': {
    title: 'Early Warning Centre',
    subtitle: 'Pollution warnings, alerts and risk trends',
  },
  '/fog': {
    title: 'Fog Computing',
    subtitle: 'Edge processing, data screening and transmission analysis',
  },
  '/analytics': {
    title: 'Analytics & Reports',
    subtitle: 'In-depth analysis, correlations and system reports',
  },
  '/settings': {
    title: 'Settings',
    subtitle: 'System configuration, data management and model provenance',
  },
};

export const Header: React.FC<HeaderProps> = ({ onRefresh, isRefreshing }) => {
  const location = useLocation();
  const [formattedTime, setFormattedTime] = useState<string>('Jan 27, 2025 14:32');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      const mon = monthNames[now.getMonth()];
      const day = now.getDate();
      const yr = now.getFullYear();
      const hrs = String(now.getHours()).padStart(2, '0');
      const mins = String(now.getMinutes()).padStart(2, '0');
      setFormattedTime(`${mon} ${day}, ${yr} ${hrs}:${mins}`);
    };
    updateTime();
    const timer = setInterval(updateTime, 30000);
    return () => clearInterval(timer);
  }, []);

  const currentInfo = pageTitles[location.pathname] || {
    title: 'AeroSense AI',
    subtitle: 'Air Quality Intelligence Platform',
  };

  return (
    <header className="h-16 bg-[#080F1D] border-b border-[#1E314B] px-8 flex items-center justify-between sticky top-0 z-20">
      <div>
        <h2 className="text-base font-bold text-white tracking-tight">{currentInfo.title}</h2>
        <p className="text-xs text-slate-400 font-normal">{currentInfo.subtitle}</p>
      </div>

      <div className="flex items-center space-x-3">
        {/* Date/Time capsule */}
        <div className="px-3.5 py-1.5 rounded-lg bg-[#142238] border border-[#1E314B] text-xs font-mono text-slate-300">
          {formattedTime}
        </div>

        {/* Refresh Button */}
        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="w-8 h-8 rounded-lg bg-[#142238] hover:bg-[#1B2B42] text-slate-300 border border-[#1E314B] flex items-center justify-center transition-colors disabled:opacity-50 cursor-pointer"
            title="Refresh current data"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
          </button>
        )}

        {/* Bell notification button */}
        <button
          className="relative w-8 h-8 rounded-lg bg-[#142238] hover:bg-[#1B2B42] text-slate-300 border border-[#1E314B] flex items-center justify-center transition-colors cursor-pointer"
          title="Notifications"
        >
          <Bell className="w-4 h-4 text-slate-300" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-rose-500 border border-[#080F1D]" />
        </button>

        {/* User avatar circle */}
        <div className="w-8 h-8 rounded-full bg-[#1B2B42] border border-teal-accent/40 flex items-center justify-center text-xs font-bold text-teal-accent shadow-xs">
          A
        </div>
      </div>
    </header>
  );
};
