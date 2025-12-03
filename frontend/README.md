# Threat Alert Dashboard - Frontend

React + TypeScript frontend for the Threat Alert Dashboard.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

The app will be available at http://localhost:5173

## Build

To build for production:
```bash
npm run build
```

## Environment Variables

Create a `.env` file (see `.env.example`):
```
VITE_API_URL=http://localhost:8000
```

## Features

- **Threat Cards**: Visual display of threat alerts with severity badges
- **Interactive Map**: Leaflet map with color-coded markers
- **Statistics Dashboard**: Real-time metrics cards
- **Detail Views**: Modal with full threat details
- **Filtering**: Filter by severity and data source
- **State Management**: Zustand for global state

## Tech Stack

- React 18
- TypeScript
- Vite
- Zustand (state management)
- Leaflet (maps)
- Axios (HTTP client)


