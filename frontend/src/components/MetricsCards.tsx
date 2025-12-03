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
          title="Last 24 Hours"
          value={statistics.recent_24h.toString()}
          icon="⏰"
        />
        <MetricCard
          title="High Priority"
          value={statistics.high_priority.toString()}
          icon="⚠️"
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


