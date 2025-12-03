/**
 * ThreatMap component - displays threats on a map using Leaflet
 */

import React from 'react';
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
  // Filter alerts with valid coordinates
  // Valid coordinates: not null, numbers, within valid ranges
  const alertsWithLocation = alerts.filter(a => {
    const lat = a.lat;
    const lon = a.lon;
    return (
      lat !== null && 
      lon !== null && 
      typeof lat === 'number' && 
      typeof lon === 'number' &&
      !isNaN(lat) && 
      !isNaN(lon) &&
      lat >= -90 && lat <= 90 &&
      lon >= -180 && lon <= 180
    );
  });
  
  if (alertsWithLocation.length === 0) {
    return (
      <div className="map-container empty">
        <p>No alerts with valid location coordinates to display</p>
      </div>
    );
  }
  
  // Calculate center point from valid coordinates
  const validLats = alertsWithLocation.map(a => a.lat!).filter(lat => !isNaN(lat));
  const validLons = alertsWithLocation.map(a => a.lon!).filter(lon => !isNaN(lon));
  
  const centerLat = validLats.reduce((sum, lat) => sum + lat, 0) / validLats.length;
  const centerLon = validLons.reduce((sum, lon) => sum + lon, 0) / validLons.length;
  
  // Calculate appropriate zoom level based on spread of points
  const latRange = Math.max(...validLats) - Math.min(...validLats);
  const lonRange = Math.max(...validLons) - Math.min(...validLons);
  const maxRange = Math.max(latRange, lonRange);
  
  // Determine zoom level (rough calculation)
  let zoom = 12;
  if (maxRange > 10) zoom = 6;
  else if (maxRange > 5) zoom = 8;
  else if (maxRange > 1) zoom = 10;
  else if (maxRange > 0.1) zoom = 12;
  else zoom = 14;
  
  return (
    <div className="map-container">
      <MapContainer
        center={[centerLat, centerLon]}
        zoom={zoom}
        style={{ height: '500px', width: '100%' }}
        key={`map-${alertsWithLocation.length}-${centerLat}-${centerLon}`}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {alertsWithLocation.map((alert) => {
          const severityLabel = alert.severity !== null ? numericToSeverity(alert.severity) : 'Unknown';
          const color = getSeverityColor(severityLabel);
          const radius = alert.severity && alert.severity >= 8 ? 10 : alert.severity && alert.severity >= 5 ? 8 : 6;
          
          // Ensure coordinates are valid numbers
          const lat = typeof alert.lat === 'number' && !isNaN(alert.lat) ? alert.lat : 0;
          const lon = typeof alert.lon === 'number' && !isNaN(alert.lon) ? alert.lon : 0;
          
          return (
            <CircleMarker
              key={alert.id}
              center={[lat, lon]}
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
                  <p><strong>Coordinates:</strong> {lat.toFixed(6)}, {lon.toFixed(6)}</p>
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


