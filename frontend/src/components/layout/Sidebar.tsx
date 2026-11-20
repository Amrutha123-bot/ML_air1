import React, { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  MapPin,
  TrendingUp,
  AlertTriangle,
  Cpu,
  Radio,
  BarChart3,
  Settings,
  Activity,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import { apiService } from '../../api/client';
import type { HealthResponse } from '../../types';

interface NavItem {
  name: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
}

const navItems: NavItem[] = [
  { name: 'Overview', path: '/', icon: LayoutDashboard },
  { name: 'Monitoring Stations', path: '/stations', icon: MapPin },
  { name: 'AI Forecasting', path: '/forecasting', icon: TrendingUp },
  { name: 'Model Laboratory', path: '/models', icon: Cpu },
  { name: 'Early Warning Centre', path: '/warnings', icon: AlertTriangle },
  { name: 'Fog Computing', path: '/fog', icon: Radio },
  { name: 'Analytics & Reports', path: '/analytics', icon: BarChart3 },
  { name: 'Settings', path: '/settings', icon: Settings },
];

export const Sidebar: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(false);

  useEffect(() => {
    let mounted = true;
    const checkHealth = async () => {
      try {
        const data = await apiService.getHealth();
        if (mounted) {
          setHealth(data);
          setIsOnline(true);
        }
      } catch {
        if (mounted) {
          setIsOnline(false);
        }
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <aside className="w-64 bg-[#0B1424] text-slate-300 flex flex-col h-screen fixed left-0 top-0 z-30 border-r border-[#1E314B] select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-[#1E314B] flex items-center space-x-3">
        <div className="w-9 h-9 rounded-lg bg-teal-accent flex items-center justify-center shadow-lg shadow-teal-500/20">
          <Activity className="w-5 h-5 text-[#080F1D]" />
        </div>
        <div>
          <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-1.5">
            AeroSense AI
          </h1>
          <p className="text-[11px] text-slate-400">Air Quality Intelligence Platform</p>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center justify-between px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-[#14D6C5] text-[#080F1D] font-bold shadow-md shadow-teal-500/20'
                    : 'text-slate-400 hover:bg-[#142238] hover:text-white'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <div className="flex items-center space-x-3">
                    <Icon className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-[#080F1D]' : 'text-slate-400'}`} />
                    <span>{item.name}</span>
                  </div>
                  {item.badge && (
                    <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-slate-800 text-teal-accent border border-teal-accent/30">
                      {item.badge}
                    </span>
                  )}
                </>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Backend Engine Status Pill */}
      <div className="p-4 border-t border-[#1E314B] bg-[#080F1D]/50">
        <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-[#142238] border border-[#1E314B]">
          <div className="flex items-center space-x-2">
            <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-[#14D6C5] animate-pulse' : 'bg-rose-500'}`} />
            <span className="text-xs font-semibold text-slate-200">
              {isOnline ? 'Backend: Connected' : 'Backend: Disconnected'}
            </span>
          </div>
          <span className="text-[10px] font-mono text-slate-400">v2.0.0</span>
        </div>
      </div>
    </aside>
  );
};
