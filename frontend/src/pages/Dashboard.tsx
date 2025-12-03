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
  
  return (
    <div className="app">
      <header className="app-header">
        <h1>🚨 Sentinel Swahili</h1>
        <p>Swahili-first threat intelligence and monitoring dashboard</p>
      </header>
      
      <div className="app-container">
        {statistics && <MetricsCards statistics={statistics} />}
        
        <div className="main-content">
          <aside className="sidebar">
            <section className="sidebar-section">
              <h3>🔍 Filters & Alerts</h3>
              
              <div className="filter-group">
                <label htmlFor="severity-slider">Minimum Severity Level</label>
                <input
                  id="severity-slider"
                  type="range"
                  min="0"
                  max="10"
                  value={minSeverity}
                  onChange={(e) => setMinSeverity(Number(e.target.value))}
                  className="slider"
                />
                <span className="slider-value">{minSeverity}</span>
              </div>
              
              <div className="filter-group">
                <label htmlFor="data-source">Data Source</label>
                <select
                  id="data-source"
                  value={dataSource}
                  onChange={(e) => setDataSource(e.target.value as DataSource)}
                  className="select"
                >
                  <option value="Agent Reports">Agent Reports</option>
                  <option value="Database Alerts">Database Alerts</option>
                  <option value="Both">Both</option>
                </select>
              </div>
              
              <div className="sidebar-stats">
                <h4>📈 Statistics</h4>
                {statistics && (
                  <>
                    <div className="stat-item">
                      <span>Total Alerts:</span>
                      <strong>{statistics.total_alerts}</strong>
                    </div>
                    <div className="stat-item">
                      <span>Filtered:</span>
                      <strong>{alerts.length}</strong>
                    </div>
                    <div className="stat-item">
                      <span>Agent Reports:</span>
                      <strong>{statistics.agent_reports_count}</strong>
                    </div>
                    <div className="stat-item">
                      <span>DB Alerts:</span>
                      <strong>{statistics.db_alerts_count}</strong>
                    </div>
                  </>
                )}
              </div>
              
              <div className="sidebar-section">
                <h4>🔬 Evaluation</h4>
                <button
                  onClick={handleRunEvaluation}
                  disabled={evaluating}
                  className="evaluation-button"
                >
                  {evaluating ? 'Running...' : 'Run Evaluation'}
                </button>
                {evaluationResult && (
                  <div className="evaluation-result">
                    <pre>{JSON.stringify(evaluationResult, null, 2)}</pre>
                  </div>
                )}
              </div>
            </section>
          </aside>
          
          <main className="content-area">
            <div className="alerts-section">
              <h3>Threat Alerts</h3>
              <p className="alerts-count">Showing {alerts.length} alert(s) - Click on a card to view details</p>
              
              {loading && <div className="loading">Loading alerts...</div>}
              {error && <div className="error">Error: {error}</div>}
              
              {!loading && !error && alerts.length === 0 && (
                <div className="empty-state">
                  <p>No alerts match the current filter criteria.</p>
                </div>
              )}
              
              <div className="alerts-list">
                {alerts.map((alert) => (
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

