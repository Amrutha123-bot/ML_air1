import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { Overview } from './pages/Overview';
import { MonitoringStations } from './pages/MonitoringStations';
import { AIForecasting } from './pages/AIForecasting';
import { EarlyWarningCentre } from './pages/EarlyWarningCentre';
import { ModelLaboratory } from './pages/ModelLaboratory';
import { FogComputing } from './pages/FogComputing';
import { AnalyticsReports } from './pages/AnalyticsReports';
import { Settings } from './pages/Settings';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="stations" element={<MonitoringStations />} />
          <Route path="forecasting" element={<AIForecasting />} />
          <Route path="warnings" element={<EarlyWarningCentre />} />
          <Route path="models" element={<ModelLaboratory />} />
          <Route path="fog" element={<FogComputing />} />
          <Route path="analytics" element={<AnalyticsReports />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
