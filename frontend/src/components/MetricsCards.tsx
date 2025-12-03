/**
 * MetricsCards component - displays dashboard statistics
 */

import React from 'react';
import type { Statistics } from '../types';
import './MetricsCards.css';

interface MetricsCardsProps {
  statistics: Statistics;
}

export const MetricsCards: React.FC<MetricsCardsProps> = ({ statistics }) => {
  const highAlerts = statistics.severity_distribution?.High || 0;
  const mediumAlerts = statistics.severity_distribution?.Medium || 0;
  const lowAlerts = statistics.severity_distribution?.Low || 0;

  return (
    <div className="metrics-container">
      <div className="metrics-header">
        <h2>📈 Overview</h2>
      </div>
      <div className="metrics-grid">
        <MetricCard
          title="Total Alerts"
          value={statistics.total_alerts.toString()}
          icon="📊"
        />
        <MetricCard
          title="Critical Alerts"
          value={statistics.critical_alerts.toString()}
          icon="🔴"
        />
        <MetricCard
          title="High Alerts"
          value={highAlerts.toString()}
          icon="⚠️"
        />
        <MetricCard
          title="Medium Alerts"
          value={mediumAlerts.toString()}
          icon="🟡"
        />
        <MetricCard
          title="Low Alerts"
          value={lowAlerts.toString()}
          icon="🟢"
        />
      </div>
    </div>
  );
};

interface MetricCardProps {
  title: string;
  value: string;
  icon?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, icon }) => {
  return (
    <div className="metric-card">
      {icon && <div className="metric-icon">{icon}</div>}
      <div className="metric-title">{title}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
};


