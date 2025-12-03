/**
 * Main App component with routing
 */

import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Dashboard } from './pages/Dashboard';
import { ThreatDetailPage } from './pages/ThreatDetailPage';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/alert/:alertId" element={<ThreatDetailPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;


