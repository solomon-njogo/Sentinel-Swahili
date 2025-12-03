/**
 * SingleAlertMap component - displays a single alert location on a map
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

interface SingleAlertMapProps {
  alert: Alert;
}

export const SingleAlertMap: React.FC<SingleAlertMapProps> = ({ alert }) => {
  const lat = alert.lat;
  const lon = alert.lon;
  
  // Validate coordinates
  const hasValidCoords = (
    lat !== null && 
    lon !== null && 
    typeof lat === 'number' && 
    typeof lon === 'number' &&
    !isNaN(lat) && 
    !isNaN(lon) &&
    lat >= -90 && lat <= 90 &&
    lon >= -180 && lon <= 180
  );
  
  if (!hasValidCoords) {
    return (
      <div className="map-container empty">
        <p>No valid location coordinates available for this alert</p>
      </div>
    );
  }
  
  const severityLabel = alert.severity !== null ? numericToSeverity(alert.severity) : 'Unknown';
  const color = getSeverityColor(severityLabel);
  const radius = alert.severity && alert.severity >= 8 ? 12 : alert.severity && alert.severity >= 5 ? 10 : 8;
  
  return (
    <div className="map-container">
      <MapContainer
        center={[lat!, lon!]}
        zoom={14}
        style={{ height: '500px', width: '100%' }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <CircleMarker
          center={[lat!, lon!]}
          radius={radius}
          pathOptions={{
            color: color,
            fillColor: color,
            fillOpacity: 0.8,
            weight: 3,
          }}
        >
          <Popup>
            <div className="map-popup">
              <h4 style={{ color, marginBottom: '10px', borderBottom: `2px solid ${color}`, paddingBottom: '5px' }}>
                {severityLabel} Alert
              </h4>
              <p><strong>ID:</strong> {alert.id}</p>
              <p><strong>Severity:</strong> {alert.severity?.toFixed(1) || 'N/A'}/10</p>
              <p><strong>Coordinates:</strong> {lat!.toFixed(6)}, {lon!.toFixed(6)}</p>
              <p style={{ fontSize: '0.9em', color: '#666' }}>
                {alert.text.substring(0, 150)}{alert.text.length > 150 ? '...' : ''}
              </p>
              {alert.source && <p><strong>Source:</strong> {alert.source}</p>}
            </div>
          </Popup>
        </CircleMarker>
      </MapContainer>
    </div>
  );
};

