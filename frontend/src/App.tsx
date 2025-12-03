/**
 * Main App component with routing
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Dashboard } from './pages/Dashboard';
import { ThreatDetailPage } from './pages/ThreatDetailPage';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <div className="app-root">
        <div className="app-root-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/alert/:alertId" element={<ThreatDetailPage />} />
          </Routes>
        </div>
        <footer className="app-footer">
          <span>Developed by <strong>Solomon Njogo</strong> and <strong>Lewis Mwangi</strong></span>
        </footer>
      </div>
    </BrowserRouter>
  );
}

export default App;


