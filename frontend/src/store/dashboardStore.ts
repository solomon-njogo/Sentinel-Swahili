/**
 * Zustand store for dashboard state management
 */

import { create } from 'zustand';
import type { Alert, DataSource } from '../types';

interface DashboardState {
  selectedAlertId: string | null;
  minSeverity: number;
  dataSource: DataSource;
  alerts: Alert[];
  loading: boolean;
  error: string | null;
  
  setSelectedAlertId: (id: string | null) => void;
  setMinSeverity: (severity: number) => void;
  setDataSource: (source: DataSource) => void;
  setAlerts: (alerts: Alert[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  selectedAlertId: null,
  minSeverity: 3,
  dataSource: 'Both',
  alerts: [],
  loading: false,
  error: null,
  
  setSelectedAlertId: (id) => set({ selectedAlertId: id }),
  setMinSeverity: (severity) => set({ minSeverity: severity }),
  setDataSource: (source) => set({ dataSource: source }),
  setAlerts: (alerts) => set({ alerts }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
}));


