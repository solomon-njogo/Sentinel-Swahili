/**
 * API client service for communicating with FastAPI backend
 */

import axios from 'axios';
import type { Alert, AgentReport, Statistics, EvaluationResponse, DataSource } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const alertsApi = {
  /**
   * Get all alerts with optional filtering
   */
  getAlerts: async (
    minSeverity: number = 0,
    dataSource: DataSource = 'Both'
  ): Promise<Alert[]> => {
    const response = await api.get<Alert[]>('/api/alerts', {
      params: {
        min_severity: minSeverity,
        data_source: dataSource,
      },
    });
    return response.data;
  },

  /**
   * Get a single alert by ID
   */
  getAlert: async (alertId: string): Promise<Alert> => {
    const response = await api.get<Alert>(`/api/alerts/${alertId}`);
    return response.data;
  },
};

export const reportsApi = {
  /**
   * Get all agent reports
   */
  getAgentReports: async (): Promise<AgentReport[]> => {
    const response = await api.get<AgentReport[]>('/api/agent-reports');
    return response.data;
  },

  /**
   * Get a single agent report by ID
   */
  getAgentReport: async (reportId: string): Promise<AgentReport> => {
    const response = await api.get<AgentReport>(`/api/agent-reports/${reportId}`);
    return response.data;
  },
};

export const statisticsApi = {
  /**
   * Get dashboard statistics
   */
  getStatistics: async (dataSource: DataSource = 'Both'): Promise<Statistics> => {
    const response = await api.get<Statistics>('/api/statistics', {
      params: {
        data_source: dataSource,
      },
    });
    return response.data;
  },
};

export const evaluationApi = {
  /**
   * Run evaluation and get report
   */
  runEvaluation: async (): Promise<EvaluationResponse> => {
    const response = await api.post<EvaluationResponse>('/api/evaluation');
    return response.data;
  },
};

export default api;


