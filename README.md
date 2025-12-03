# Threat Alert Dashboard

A modern threat intelligence and monitoring dashboard with a React + TypeScript frontend and FastAPI Python backend.

## Architecture

- **Frontend**: React 18 + TypeScript + Vite
- **Backend**: FastAPI (Python)
- **Communication**: REST API with JSON

## Project Structure

```
.
├── backend/           # FastAPI backend
│   ├── api/          # API route handlers
│   ├── models/       # Pydantic models
│   ├── services/     # Business logic
│   └── main.py       # FastAPI app entry point
├── frontend/         # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── services/    # API client
│   │   ├── store/       # State management
│   │   ├── types/       # TypeScript types
│   │   └── utils/       # Utility functions
│   └── package.json
└── dashboard/        # Original Streamlit dashboard (legacy)
```

## Setup Instructions

### Backend Setup

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Run the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Features

- **Threat Cards**: Visual card-based display of threat alerts
- **Interactive Map**: Leaflet map showing threat locations
- **Statistics Dashboard**: Real-time metrics and charts
- **Detail Views**: Expandable threat details with validation/escalation data
- **Filtering**: Filter by severity level and data source
- **Evaluation**: Run evaluation reports on threat data

## API Endpoints

- `GET /api/alerts` - Get all alerts (with filtering)
- `GET /api/alerts/{alert_id}` - Get single alert
- `GET /api/agent-reports` - Get all agent reports
- `GET /api/agent-reports/{report_id}` - Get single agent report
- `GET /api/statistics` - Get dashboard statistics
- `POST /api/evaluation` - Run evaluation

## Development

### Backend
- Uses existing dashboard functions from `dashboard/` directory
- Reuses agent loader and geocoder utilities
- Maintains compatibility with existing data sources

### Frontend
- TypeScript for type safety
- Zustand for state management
- React hooks for data fetching
- Responsive design with CSS variables

## Notes

- The original Streamlit dashboard in `dashboard/` is kept for reference
- Data sources: SQLite database, agent reports JSON files, CSV files
- CORS is configured for local development
