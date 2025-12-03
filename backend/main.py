#!/usr/bin/env python3
"""
FastAPI Backend for Threat Alert Dashboard
Provides REST API endpoints for the React frontend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from backend.api import alerts, reports, statistics, evaluation

app = FastAPI(
    title="Threat Alert Dashboard API",
    description="REST API for Threat Alert Dashboard",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(alerts.router, prefix="/api", tags=["alerts"])
app.include_router(reports.router, prefix="/api", tags=["reports"])
app.include_router(statistics.router, prefix="/api", tags=["statistics"])
app.include_router(evaluation.router, prefix="/api", tags=["evaluation"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Threat Alert Dashboard API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


