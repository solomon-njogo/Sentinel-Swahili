# Threat Alert Dashboard - Backend API

FastAPI backend for the Threat Alert Dashboard.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn main:app --reload --port 8000
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

### Alerts
- `GET /api/alerts` - Get all alerts
  - Query params: `min_severity` (0-10), `data_source` (Agent Reports/Database Alerts/Both)
- `GET /api/alerts/{alert_id}` - Get single alert

### Agent Reports
- `GET /api/agent-reports` - Get all agent reports
- `GET /api/agent-reports/{report_id}` - Get single agent report

### Statistics
- `GET /api/statistics` - Get dashboard statistics
  - Query params: `data_source` (Agent Reports/Database Alerts/Both)

### Evaluation
- `POST /api/evaluation` - Run evaluation and get report

## Data Sources

The backend reads from:
- SQLite database: `data/threats.db`
- Agent reports: `agents/storage/*.json`
- Processed reports: `data/processed_reports.csv`


