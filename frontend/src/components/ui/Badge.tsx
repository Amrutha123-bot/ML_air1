import React from 'react';
import { getSeverityStyle } from '../../utils/severity';

interface BadgeProps {
  label: string;
  color?: string;
  showDot?: boolean;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  label,
  color,
  showDot = true,
  className = '',
}) => {
  const style = getSeverityStyle(label, color);

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold border ${className}`}
      style={{
        backgroundColor: style.bg,
        color: style.text,
        borderColor: style.border,
      }}
    >
      {showDot && (
        <span
          className="w-1.5 h-1.5 rounded-full mr-1.5 flex-shrink-0"
          style={{ backgroundColor: style.dot }}
        />
      )}
      {label}
    </span>
  );
};

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ComponentType<{ className?: string }>;
  trend?: {
    value: string;
    isPositive?: boolean;
  };
  colorClass?: string;
  badge?: React.ReactNode;
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  colorClass = 'from-teal-500 to-teal-700',
  badge,
}) => {
  return (
    <div className="bg-[#142238] rounded-xl border border-[#1E314B] p-5 shadow-md flex items-center justify-between">
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">{title}</p>
          {badge}
        </div>
        <p className="text-2xl font-extrabold text-white tracking-tight font-mono">{value}</p>
        {(subtitle || trend) && (
          <div className="flex items-center space-x-2 text-xs">
            {trend && (
              <span className={`font-semibold ${trend.isPositive ? 'text-teal-400' : 'text-rose-400'}`}>
                {trend.value}
              </span>
            )}
            {subtitle && <span className="text-slate-400">{subtitle}</span>}
          </div>
        )}
      </div>
      {Icon && (
        <div className={`w-11 h-11 rounded-lg bg-gradient-to-br ${colorClass} flex items-center justify-center text-white shadow-md shadow-teal-900/20`}>
          <Icon className="w-5 h-5" />
        </div>
      )}
    </div>
  );
};
