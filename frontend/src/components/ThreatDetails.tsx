/**
 * ThreatDetails component - displays detailed information about a threat
 */

import React, { useState, useEffect } from 'react';
import type { Alert, AgentReport } from '../types';
import { alertsApi, reportsApi } from '../services/api';
import { numericToSeverity, getSeverityColor } from '../utils/severity';
import { StructuredDataView } from './StructuredDataView';
import './ThreatDetails.css';

interface ThreatDetailsProps {
  alert: Alert;
  onClose: () => void;
}

export const ThreatDetails: React.FC<ThreatDetailsProps> = ({ alert, onClose }) => {
  const [agentReport, setAgentReport] = useState<AgentReport | null>(null);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    if (alert.source === 'agent') {
      setLoading(true);
      reportsApi.getAgentReport(alert.id)
        .then(setAgentReport)
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [alert]);
  
  const severityLabel = alert.severity !== null ? numericToSeverity(alert.severity) : 'Unknown';
  const severityColor = getSeverityColor(severityLabel);
  
  return (
    <div className="threat-details-overlay" onClick={onClose}>
      <div className="threat-details-modal" onClick={(e) => e.stopPropagation()}>
        <div className="threat-details-header">
          <h3>📋 Alert: {alert.id}</h3>
          <button className="close-button" onClick={onClose}>×</button>
        </div>
        
        <div className="threat-details-content">
          <section className="details-section">
            <h4>📊 Basic Information</h4>
            <div className="details-grid">
              <div className="detail-item">
                <span 
                  className="severity-badge"
                  style={{ backgroundColor: severityColor }}
                >
                  {severityLabel} ({alert.severity?.toFixed(1) || 'N/A'})
                </span>
              </div>
              <div className="detail-item">
                <strong>Source:</strong> {alert.source ? alert.source.charAt(0).toUpperCase() + alert.source.slice(1) : 'N/A'}
              </div>
              {alert.received_at && (
                <div className="detail-item">
                  <strong>Received At:</strong> {alert.received_at}
                </div>
              )}
            </div>
          </section>
          
          {alert.lat !== null && alert.lon !== null && (
            <section className="details-section">
              <h4>📍 Location</h4>
              <p><strong>Coordinates:</strong> {alert.lat.toFixed(6)}, {alert.lon.toFixed(6)}</p>
            </section>
          )}
          
          <section className="details-section">
            <h4>💬 Full Details</h4>
            <div className="info-box">{alert.text}</div>
          </section>
          
          {agentReport && (
            <>
              {agentReport.validation && (
                <section className="details-section">
                  <h4>✅ Validation Results</h4>
                  <StructuredDataView data={agentReport.validation} />
                </section>
              )}
              
              {agentReport.escalation && (
                <section className="details-section">
                  <h4>🚨 Escalation Results</h4>
                  <StructuredDataView data={agentReport.escalation} />
                </section>
              )}
            </>
          )}
          
          {(alert.validation || alert.escalation || alert.classification) && (
            <section className="details-section">
              <h4>🔍 Classification Data</h4>
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


