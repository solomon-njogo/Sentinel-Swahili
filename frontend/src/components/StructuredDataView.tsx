/**
 * StructuredDataView component - displays validation/escalation/classification data
 * in readable format with JSON toggle
 */

import React, { useState } from 'react';
import './StructuredDataView.css';

interface StructuredDataViewProps {
  data: Record<string, any>;
  title?: string;
}

export const StructuredDataView: React.FC<StructuredDataViewProps> = ({ data, title }) => {
  const [showJson, setShowJson] = useState(false);

  if (!data || Object.keys(data).length === 0) {
    return null;
  }

  const formatReadableData = (obj: Record<string, any>): JSX.Element => {
    const formatValue = (value: any): string => {
      if (value === null || value === undefined) return 'N/A';
      if (typeof value === 'boolean') return value ? 'Yes' : 'No';
      if (typeof value === 'number') {
        if (value < 1 && value > 0) return `${(value * 100).toFixed(1)}%`;
        return value.toString();
      }
      if (Array.isArray(value)) {
        return value.length > 0 ? value.join(', ') : 'None';
      }
      if (typeof value === 'object') {
        return JSON.stringify(value, null, 2);
      }
      return String(value);
    };

    const formatFieldName = (key: string): string => {
      return key
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
    };

    const renderField = (key: string, value: any) => {
      const formattedKey = formatFieldName(key);
      
      // Special formatting for specific fields
      if (key === 'severity' && typeof value === 'string') {
        return (
          <div key={key} className="data-field">
            <span className="field-label">{formattedKey}:</span>
            <span className="field-value severity-value">{value}</span>
          </div>
        );
      }
      
      if (key === 'status' && typeof value === 'string') {
        const statusColor = value.toLowerCase() === 'valid' ? '#388E3C' : 
                           value.toLowerCase() === 'invalid' ? '#D32F2F' : '#F57C00';
        return (
          <div key={key} className="data-field">
            <span className="field-label">{formattedKey}:</span>
            <span className="field-value" style={{ color: statusColor, fontWeight: 600 }}>
              {value}
            </span>
          </div>
        );
      }
      
      if (key === 'requires_immediate_alert' && typeof value === 'boolean') {
        return (
          <div key={key} className="data-field">
            <span className="field-label">{formattedKey}:</span>
            <span className="field-value" style={{ 
              color: value ? '#D32F2F' : '#388E3C',
              fontWeight: 600 
            }}>
              {value ? '⚠️ Yes' : 'No'}
            </span>
          </div>
        );
      }
      
      if (key === 'escalation_window_minutes' && typeof value === 'number') {
        let windowDisplay = '';
        if (value < 60) {
          windowDisplay = `${value} minute${value !== 1 ? 's' : ''}`;
        } else if (value < 1440) {
          const hours = Math.floor(value / 60);
          windowDisplay = `${hours} hour${hours !== 1 ? 's' : ''}`;
        } else {
          const days = Math.floor(value / 1440);
          windowDisplay = `${days} day${days !== 1 ? 's' : ''}`;
        }
        return (
          <div key={key} className="data-field">
            <span className="field-label">{formattedKey}:</span>
            <span className="field-value">{windowDisplay}</span>
          </div>
        );
      }
      
      if (key === 'priority_score' && typeof value === 'number') {
        return (
          <div key={key} className="data-field">
            <span className="field-label">{formattedKey}:</span>
            <span className="field-value">{value.toFixed(2)}</span>
          </div>
        );
      }
      
      if (key === 'confidence' || key === 'classification_confidence') {
        const percentage = typeof value === 'number' ? (value * 100).toFixed(1) : formatValue(value);
        return (
          <div key={key} className="data-field">
            <span className="field-label">{formattedKey}:</span>
            <span className="field-value">{percentage}%</span>
          </div>
        );
      }
      
      if (key === 'urgency_keywords_found' && Array.isArray(value)) {
        return (
          <div key={key} className="data-field">
            <span className="field-label">{formattedKey}:</span>
            <div className="field-value">
              {value.length > 0 ? (
                <div className="keywords-list">
                  {value.map((keyword, idx) => (
                    <span key={idx} className="keyword-tag">{keyword}</span>
                  ))}
                </div>
              ) : (
                <span>None found</span>
              )}
            </div>
          </div>
        );
      }
      
      // Handle nested objects (like extracted_entities)
      if (typeof value === 'object' && !Array.isArray(value) && value !== null) {
        return (
          <div key={key} className="data-field nested">
            <span className="field-label">{formattedKey}:</span>
            <div className="nested-object">
              {Object.entries(value).map(([nestedKey, nestedValue]) => (
                <div key={nestedKey} className="nested-field">
                  <span className="nested-label">{formatFieldName(nestedKey)}:</span>
                  <span className="nested-value">{formatValue(nestedValue)}</span>
                </div>
              ))}
            </div>
          </div>
        );
      }
      
      return (
        <div key={key} className="data-field">
          <span className="field-label">{formattedKey}:</span>
          <span className="field-value">{formatValue(value)}</span>
        </div>
      );
    };

    return (
      <div className="readable-data-view">
        {Object.entries(obj).map(([key, value]) => renderField(key, value))}
      </div>
    );
  };

  return (
    <div className="structured-data-view">
      <div className="data-view-header">
        {title && <h3>{title}</h3>}
        <label className="json-toggle">
          <input
            type="checkbox"
            checked={showJson}
            onChange={(e) => setShowJson(e.target.checked)}
          />
          <span>Show JSON</span>
        </label>
      </div>
      <div className="data-view-content">
        {showJson ? (
          <div className="json-view">
            <pre>{JSON.stringify(data, null, 2)}</pre>
          </div>
        ) : (
          formatReadableData(data)
        )}
      </div>
    </div>
  );
};

