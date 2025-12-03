/**
 * Dashboard page - main threat alerts view
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAlerts, useStatistics } from '../hooks/useAlerts';
import { useDashboardStore } from '../store/dashboardStore';
import { MetricsCards } from '../components/MetricsCards';
import { ThreatCard } from '../components/ThreatCard';
import { ThreatMap } from '../components/ThreatMap';
import { evaluationApi } from '../services/api';
import type { DataSource } from '../types';
import '../App.css';

type SortOption = 'severity-desc' | 'severity-asc' | 'date-desc' | 'date-asc' | 'id-asc' | 'id-desc';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const { alerts, loading, error } = useAlerts();
  const { statistics } = useStatistics();
  const {
    minSeverity,
    dataSource,
    setMinSeverity,
    setDataSource,
  } = useDashboardStore();
  
  const [evaluationResult, setEvaluationResult] = useState<any>(null);
  const [evaluating, setEvaluating] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortOption, setSortOption] = useState<SortOption>('severity-desc');
  
  const handleRunEvaluation = async () => {
    setEvaluating(true);
    try {
      const result = await evaluationApi.runEvaluation();
      setEvaluationResult(result);
    } catch (err) {
      console.error('Evaluation failed:', err);
    } finally {
      setEvaluating(false);
    }
  };
  
  const handleCardClick = (alertId: string) => {
    navigate(`/alert/${alertId}`);
  };

  // Filter and sort alerts
  const filteredAndSortedAlerts = React.useMemo(() => {
    let filtered = [...alerts];

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(alert => 
        alert.id.toLowerCase().includes(query) ||
        alert.text?.toLowerCase().includes(query) ||
        alert.source?.toLowerCase().includes(query)
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortOption) {
        case 'severity-desc':
          return (b.severity ?? 0) - (a.severity ?? 0);
        case 'severity-asc':
          return (a.severity ?? 0) - (b.severity ?? 0);
        case 'date-desc':
          const dateA = a.received_at ? new Date(a.received_at).getTime() : 0;
          const dateB = b.received_at ? new Date(b.received_at).getTime() : 0;
          return dateB - dateA;
        case 'date-asc':
          const dateA2 = a.received_at ? new Date(a.received_at).getTime() : 0;
          const dateB2 = b.received_at ? new Date(b.received_at).getTime() : 0;
          return dateA2 - dateB2;
        case 'id-asc':
          return a.id.localeCompare(b.id);
        case 'id-desc':
          return b.id.localeCompare(a.id);
        default:
          return 0;
      }
    });

    return filtered;
  }, [alerts, searchQuery, sortOption]);
  
  return (
    <div className="app">
      <header className="app-header">
        <h1>🚨 Sentinel Swahili</h1>
        <p>Swahili-first threat intelligence and monitoring dashboard</p>
      </header>
      
      <div className="app-container">
        {statistics && <MetricsCards statistics={statistics} />}
        
        <div className="main-content main-content-full">
          <main className="content-area">
            <div className="alerts-section">
              <div className="alerts-header">
                <h3>Threat Alerts</h3>
                <div className="alerts-controls">
                  <div className="search-container">
                    <input
                      type="text"
                      placeholder="Search alerts by ID, text, or source..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="search-input"
                    />
                  </div>
                  <div className="sort-container">
                    <label htmlFor="sort-select">Sort by:</label>
                    <select
                      id="sort-select"
                      value={sortOption}
                      onChange={(e) => setSortOption(e.target.value as SortOption)}
                      className="sort-select"
                    >
                      <option value="severity-desc">Severity (High to Low)</option>
                      <option value="severity-asc">Severity (Low to High)</option>
                      <option value="date-desc">Date (Newest First)</option>
                      <option value="date-asc">Date (Oldest First)</option>
                      <option value="id-asc">ID (A-Z)</option>
                      <option value="id-desc">ID (Z-A)</option>
                    </select>
                  </div>
                </div>
              </div>
              <p className="alerts-count">
                Showing {filteredAndSortedAlerts.length} of {alerts.length} alert(s) - Click on a card to view details
              </p>
              
              {loading && <div className="loading">Loading alerts...</div>}
              {error && <div className="error">Error: {error}</div>}
              
              {!loading && !error && filteredAndSortedAlerts.length === 0 && (
                <div className="empty-state">
                  <p>No alerts match the current filter criteria.</p>
                </div>
              )}
              
              <div className="alerts-list">
                {filteredAndSortedAlerts.map((alert) => (
                  <ThreatCard
                    key={alert.id}
                    alert={alert}
                    isSelected={false}
                    onClick={() => handleCardClick(alert.id)}
                  />
                ))}
              </div>
            </div>
            
            <div className="map-section">
              <h3>🗺️ Threat Map</h3>
              {(() => {
                const alertsWithCoords = alerts.filter(a => {
                  const lat = a.lat;
                  const lon = a.lon;
                  return (
                    lat !== null && lon !== null && 
                    typeof lat === 'number' && typeof lon === 'number' &&
                    !isNaN(lat) && !isNaN(lon) &&
                    lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180
                  );
                });
                return (
                  <p className="map-info">
                    {alertsWithCoords.length} alert{alertsWithCoords.length !== 1 ? 's' : ''} with valid coordinates
                  </p>
                );
              })()}
              <ThreatMap alerts={alerts} />
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};

