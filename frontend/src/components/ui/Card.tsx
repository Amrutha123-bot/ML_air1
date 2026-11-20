import React from 'react';

interface CardProps {
  title?: React.ReactNode;
  subtitle?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  action,
  children,
  className = '',
}) => {
  return (
    <div className={`bg-[#142238] rounded-xl border border-[#1E314B] shadow-md overflow-hidden text-slate-100 ${className}`}>
      {(title || action) && (
        <div className="px-5 py-3.5 border-b border-[#1E314B]/80 flex items-center justify-between">
          <div>
            {typeof title === 'string' ? (
              <h3 className="text-sm font-bold text-white tracking-tight">{title}</h3>
            ) : (
              title
            )}
            {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
};
