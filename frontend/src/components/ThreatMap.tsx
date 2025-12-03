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

  // Build a simple "heatmap" by aggregating alerts into geographic buckets
  // This avoids extra dependencies while still visualizing density.
  const bucketed = alertsWithLocation.reduce((acc, alert) => {
    const lat = typeof alert.lat === 'number' && !isNaN(alert.lat) ? alert.lat : 0;
    const lon = typeof alert.lon === 'number' && !isNaN(alert.lon) ? alert.lon : 0;

    // Bucket by ~0.2 degrees to group nearby points
    const bucketLat = Number(lat.toFixed(1));
    const bucketLon = Number(lon.toFixed(1));
    const key = `${bucketLat}:${bucketLon}`;

    const severity = alert.severity ?? 0;

    if (!acc[key]) {
      acc[key] = {
        lat: bucketLat,
        lon: bucketLon,
        count: 0,
        maxSeverity: severity,
      };
    }

    acc[key].count += 1;
    if (severity > acc[key].maxSeverity) {
      acc[key].maxSeverity = severity;
    }

    return acc;
  }, {} as Record<string, { lat: number; lon: number; count: number; maxSeverity: number }>);

  const heatPoints = Object.values(bucketed);
  const maxCount = heatPoints.length > 0 ? Math.max(...heatPoints.map(p => p.count)) : 1;
  
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
        {heatPoints.map((point, index) => {
          const severityLabel = numericToSeverity(point.maxSeverity || 0);
          const color = getSeverityColor(severityLabel);
          const intensity = point.count / maxCount; // 0–1
          const radius = 10 + intensity * 25; // larger radius for denser areas
          const fillOpacity = 0.3 + intensity * 0.4; // more opaque for denser areas

          return (
            <CircleMarker
              key={`heat-${index}`}
              center={[point.lat, point.lon]}
              radius={radius}
              pathOptions={{
                color,
                fillColor: color,
                fillOpacity,
                weight: 1,
              }}
            >
              <Popup>
                <div className="map-popup">
                  <h4
                    style={{
                      color,
                      marginBottom: '10px',
                      borderBottom: `2px solid ${color}`,
                      paddingBottom: '5px',
                    }}
                  >
                    Threat Density
                  </h4>
                  <p><strong>Alerts in area:</strong> {point.count}</p>
                  <p><strong>Max severity in area:</strong> {point.maxSeverity.toFixed(1)}/10 ({severityLabel})</p>
                  <p><strong>Approx. center:</strong> {point.lat.toFixed(4)}, {point.lon.toFixed(4)}</p>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
};


