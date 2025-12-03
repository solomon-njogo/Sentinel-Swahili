/**
 * TypeScript type definitions for the Threat Alert Dashboard
 */

export interface Alert {
  id: string;
  text: string;
  severity: number | null;
  lat: number | null;
  lon: number | null;
  source: string | null;
  received_at: string | null;
  processed_at: string | null;
  classification?: Record<string, any>;
  validation?: Record<string, any>;
  escalation?: Record<string, any>;
  priority_score?: number | null;
  requires_immediate_alert?: boolean | null;
}

export interface AgentReport {
  report_id: string;
  raw_message: string;
  source: string | null;
  received_at: string | null;
  processed_at: string | null;
  status: string | null;
  validation?: Record<string, any>;
  escalation?: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface Statistics {
  total_alerts: number;
  critical_alerts: number;
  recent_24h: number;
  high_priority: number;
  severity_distribution: Record<string, number>;
  agent_reports_count: number;
  db_alerts_count: number;
  feedback_count: number;
  processed_count: number;
}

export interface EvaluationResponse {
  generated_at: string;
  counts: {
    processed_rows: number;
    alerts_rows: number;
    feedback_rows: number;
  };
  scores: {
    concept: number;
    methodology: number;
    technical: number;
    usability: number;
    scalability: number;
  };
  total: number;
}

export type DataSource = 'Agent Reports' | 'Database Alerts' | 'Both';

export type SeverityLevel = 'Critical' | 'High' | 'Medium' | 'Low';


