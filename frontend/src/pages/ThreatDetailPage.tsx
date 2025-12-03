/**
 * ThreatDetailPage - Full page view for a single threat alert
 */

import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import type { Alert, AgentReport } from '../types';
import { alertsApi, reportsApi } from '../services/api';
import { numericToSeverity, getSeverityColor } from '../utils/severity';
import { SingleAlertMap } from '../components/SingleAlertMap';
import { StructuredDataView } from '../components/StructuredDataView';
import './ThreatDetailPage.css';

export const ThreatDetailPage: React.FC = () => {
  const { alertId } = useParams<{ alertId: string }>();
  const navigate = useNavigate();
  const [alert, setAlert] = useState<Alert | null>(null);
  const [agentReport, setAgentReport] = useState<AgentReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    if (!alertId) {
      setError('No alert ID provided');
      setLoading(false);
      return;
    }
    
    const fetchAlert = async () => {
      try {
        setLoading(true);
        const alertData = await alertsApi.getAlert(alertId);
        setAlert(alertData);
        
        // Fetch agent report if it's from an agent source
        if (alertData.source === 'agent') {
          try {
            const report = await reportsApi.getAgentReport(alertId);
            setAgentReport(report);
          } catch (err) {
            console.error('Failed to fetch agent report:', err);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load alert');
      } finally {
        setLoading(false);
      }
    };
    
    fetchAlert();
  }, [alertId]);
  
  if (loading) {
    return (
      <div className="detail-page">
        <div className="loading-container">
          <p>Loading alert details...</p>
        </div>
      </div>
    );
  }
  
  if (error || !alert) {
    return (
      <div className="detail-page">
        <div className="error-container">
          <p>Error: {error || 'Alert not found'}</p>
          <button onClick={() => navigate('/')} className="back-button">
            ← Back to Dashboard
          </button>
        </div>
      </div>
    );
  }
  
  const severityLabel = alert.severity !== null ? numericToSeverity(alert.severity) : 'Unknown';
  const severityColor = getSeverityColor(severityLabel);
  
  return (
    <div className="detail-page">
      <header className="detail-header">
        <button onClick={() => navigate('/')} className="back-button">
          ← Back to Dashboard
        </button>
        <h1>📋 Alert Details: {alert.id}</h1>
      </header>
      
      <div className="detail-container">
        <div className="detail-main">
          {/* Basic Information Section */}
          <section className="detail-section">
            <h2>📊 Basic Information</h2>
            <div className="info-grid">
              <div className="info-card">
                <label>Severity</label>
                <span 
                  className="severity-badge-large"
                  style={{ backgroundColor: severityColor }}
                >
                  {severityLabel} ({alert.severity?.toFixed(1) || 'N/A'})
                </span>
              </div>
              <div className="info-card">
                <label>Source</label>
                <span>{alert.source ? alert.source.charAt(0).toUpperCase() + alert.source.slice(1) : 'N/A'}</span>
              </div>
              {alert.received_at && (
                <div className="info-card">
                  <label>Received At</label>
                  <span>{alert.received_at}</span>
                </div>
              )}
              {alert.processed_at && (
                <div className="info-card">
                  <label>Processed At</label>
                  <span>{alert.processed_at}</span>
                </div>
              )}
            </div>
          </section>
          
          {/* Location Section */}
          {alert.lat !== null && alert.lon !== null && (
            <section className="detail-section">
              <h2>📍 Location</h2>
              <div className="location-info">
                <p><strong>Coordinates:</strong> {alert.lat.toFixed(6)}, {alert.lon.toFixed(6)}</p>
              </div>
              <div className="map-wrapper">
                <SingleAlertMap alert={alert} />
              </div>
            </section>
          )}
          
          {/* Full Details Section */}
          <section className="detail-section">
            <h2>💬 Full Details</h2>
            <div className="text-content">
              <p>{alert.text}</p>
            </div>
          </section>
          
          {/* Agent Report Sections */}
          {agentReport && (
            <>
              {agentReport.validation && (
                <section className="detail-section">
                  <h2>✅ Validation Results</h2>
                  <StructuredDataView data={agentReport.validation} />
                </section>
              )}
              
              {agentReport.escalation && (
                <section className="detail-section">
                  <h2>🚨 Escalation Results</h2>
                  <StructuredDataView data={agentReport.escalation} />
                </section>
              )}
            </>
          )}
          
          {/* Classification Data */}
          {(alert.validation || alert.escalation || alert.classification) && (
            <section className="detail-section">
              <h2>🔍 Classification Data</h2>
              {alert.validation && (
                <StructuredDataView data={alert.validation} title="Validation" />
              )}
              {alert.escalation && (
                <StructuredDataView data={alert.escalation} title="Escalation" />
              )}
              {alert.classification && (
                <StructuredDataView data={alert.classification} title="Classification" />
              )}
            </section>
          )}
        </div>
      </div>
    </div>
  );
};

