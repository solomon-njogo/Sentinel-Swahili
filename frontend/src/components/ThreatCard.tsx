/**
 * ThreatCard component - displays individual threat alert card
 */

import React from 'react';
import type { Alert } from '../types';
import { numericToSeverity, getSeverityColor } from '../utils/severity';
import './ThreatCard.css';

interface ThreatCardProps {
  alert: Alert;
  isSelected: boolean;
  onClick: () => void;
}

export const ThreatCard: React.FC<ThreatCardProps> = ({ alert, isSelected, onClick }) => {
  const severityLabel = alert.severity !== null ? numericToSeverity(alert.severity) : 'Unknown';
  const severityColor = getSeverityColor(severityLabel);
  const severityValue = alert.severity !== null ? alert.severity.toFixed(1) : 'N/A';
  
  const borderClass = severityLabel.toLowerCase();
  const cardClass = `threat-card ${borderClass} ${isSelected ? 'selected' : ''}`;
  
  const text = alert.text || 'No description available';
  const truncatedText = text.length > 150 ? text.substring(0, 150) + '...' : text;
  
  return (
    <div className={cardClass} onClick={onClick}>
      <div className="threat-card-header">
        <h4 className="threat-card-id">{alert.id}</h4>
        <span 
          className="severity-badge"
          style={{ backgroundColor: severityColor }}
        >
          {severityLabel} ({severityValue})
        </span>
      </div>
      
      {alert.lat !== null && alert.lon !== null && (
        <div className="threat-card-location">
          <strong>📍 Location:</strong> {alert.lat.toFixed(4)}, {alert.lon.toFixed(4)}
        </div>
      )}
      
      {alert.source && (
        <div className="threat-card-source">
          <strong>Source:</strong> {alert.source.charAt(0).toUpperCase() + alert.source.slice(1)}
        </div>
      )}
      
      <div className="threat-card-details">
        <strong>Details:</strong> {truncatedText}
      </div>
    </div>
  );
};


