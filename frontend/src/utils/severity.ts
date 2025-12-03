/**
 * Utility functions for severity handling
 */

import type { SeverityLevel } from '../types';

export function numericToSeverity(severityNum: number): SeverityLevel {
  if (severityNum >= 9) {
    return 'Critical';
  } else if (severityNum >= 7) {
    return 'High';
  } else if (severityNum >= 4) {
    return 'Medium';
  } else {
    return 'Low';
  }
}

export function getSeverityColor(severity: SeverityLevel | string): string {
  const colors: Record<string, string> = {
    Critical: '#D32F2F',
    High: '#F57C00',
    Medium: '#FBC02D',
    Low: '#388E3C',
  };
  return colors[severity] || '#9CA3AF';
}


