/**
 * Custom hook for fetching and managing alerts
 */

import { useEffect, useState } from 'react';
import { alertsApi, statisticsApi } from '../services/api';
import { useDashboardStore } from '../store/dashboardStore';
import type { Statistics } from '../types';

export function useAlerts() {
  const {
    alerts,
    loading,
    error,
    minSeverity,
    dataSource,
    setAlerts,
    setLoading,
    setError,
  } = useDashboardStore();
  
  useEffect(() => {
    const fetchAlerts = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await alertsApi.getAlerts(minSeverity, dataSource);
        setAlerts(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch alerts');
      } finally {
        setLoading(false);
      }
    };
    
    fetchAlerts();
  }, [minSeverity, dataSource, setAlerts, setLoading, setError]);
  
  return { alerts, loading, error };
}

export function useStatistics() {
  const { dataSource } = useDashboardStore();
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchStatistics = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await statisticsApi.getStatistics(dataSource);
        setStatistics(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch statistics');
      } finally {
        setLoading(false);
      }
    };
    
    fetchStatistics();
  }, [dataSource]);
  
  return { statistics, loading, error };
}

