"""
Agent Report Loader
Loads and parses agent reports from JSON storage files.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


def load_agent_reports(
    storage_dir: Optional[Path] = None,
    run_number: Optional[int] = None,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Load agent reports from JSON storage directory.
    
    Args:
        storage_dir: Directory containing agent reports. Defaults to agents/storage/
        run_number: Optional run number to filter reports (e.g., 6 for agents_006)
        limit: Optional limit on number of reports to return
        
    Returns:
        List of report dictionaries
    """
    if storage_dir is None:
        # Default to agents/storage relative to project root
        base_dir = Path(__file__).parent.parent
        storage_dir = base_dir / "agents" / "storage"
    
    storage_dir = Path(storage_dir)
    if not storage_dir.exists():
        return []
    
    reports = []
    
    # Find all JSON files matching the pattern
    pattern = f"agents_{run_number:03d}_report_*.json" if run_number else "agents_*_report_*.json"
    
    for json_file in sorted(storage_dir.glob(pattern), reverse=True):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
                reports.append(report)
                
                if limit and len(reports) >= limit:
                    break
        except Exception as e:
            # Skip invalid JSON files
            continue
    
    # Also load reports with RPT- prefix (older format)
    if not run_number:
        for json_file in sorted(storage_dir.glob("RPT-*.json"), reverse=True):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    reports.append(report)
                    
                    if limit and len(reports) >= limit:
                        break
            except Exception:
                continue
    
    return reports


def get_latest_reports(storage_dir: Optional[Path] = None, count: int = 50) -> List[Dict]:
    """
    Get the most recent agent reports.
    
    Args:
        storage_dir: Directory containing agent reports
        count: Number of recent reports to return
        
    Returns:
        List of most recent report dictionaries
    """
    return load_agent_reports(storage_dir=storage_dir, limit=count)

