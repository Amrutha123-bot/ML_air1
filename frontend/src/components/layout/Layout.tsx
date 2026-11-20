import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

export const Layout: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#080F1D] text-[#F8FAFC] flex">
      {/* Fixed Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex-1 ml-60 flex flex-col min-h-screen bg-[#080F1D]">
        <Header />
        <main className="flex-1 px-8 py-4 overflow-y-auto bg-[#080F1D]">
          <Outlet />
        </main>
      </div>
    </div>
  );
};
