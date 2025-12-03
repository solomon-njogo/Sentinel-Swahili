/**
 * ThreatMap component - displays threats on a map using Leaflet
 */

import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import L from 'leaflet';
import type { Alert } from '../types';
import { numericToSeverity, getSeverityColor } from '../utils/severity';
import 'leaflet/dist/leaflet.css';
import './ThreatMap.css';

// Fix for default marker icons in React-Leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

interface ThreatMapProps {
  alerts: Alert[];
}

export const ThreatMap: React.FC<ThreatMapProps> = ({ alerts }) => {
  const alertsWithLocation = alerts.filter(a => a.lat !== null && a.lon !== null);
  
  if (alertsWithLocation.length === 0) {
    return (
      <div className="map-container empty">
        <p>No alerts with location data to display</p>
      </div>
    );
  }
  
  // Calculate center point
  const centerLat = alertsWithLocation.reduce((sum, a) => sum + (a.lat || 0), 0) / alertsWithLocation.length;
  const centerLon = alertsWithLocation.reduce((sum, a) => sum + (a.lon || 0), 0) / alertsWithLocation.length;
  
  return (
    <div className="map-container">
      <MapContainer
        center={[centerLat, centerLon]}
        zoom={12}
        style={{ height: '500px', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {alertsWithLocation.map((alert) => {
          const severityLabel = alert.severity !== null ? numericToSeverity(alert.severity) : 'Unknown';
          const color = getSeverityColor(severityLabel);
          const radius = alert.severity && alert.severity >= 8 ? 10 : alert.severity && alert.severity >= 5 ? 8 : 6;
          
          return (
            <CircleMarker
              key={alert.id}
              center={[alert.lat!, alert.lon!]}
              radius={radius}
              pathOptions={{
                color: color,
                fillColor: color,
                fillOpacity: 0.7,
                weight: 2,
              }}
            >
              <Popup>
                <div className="map-popup">
                  <h4 style={{ color, marginBottom: '10px', borderBottom: `2px solid ${color}`, paddingBottom: '5px' }}>
                    {severityLabel} Alert
                  </h4>
                  <p><strong>ID:</strong> {alert.id}</p>
                  <p><strong>Severity:</strong> {alert.severity?.toFixed(1) || 'N/A'}/10</p>
                  <p style={{ fontSize: '0.9em', color: '#666' }}>
                    {alert.text.substring(0, 150)}{alert.text.length > 150 ? '...' : ''}
                  </p>
                  {alert.source && <p><strong>Source:</strong> {alert.source}</p>}
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
};


