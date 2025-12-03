"""
Storage layer for processed threat reports.
Saves reports as JSON files with run-based naming to tie them to log files.
Naming format: agents_NNN_report_MMM.json (matches log file agents_NNN.log)
"""

import json
import uuid
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from agents.data_models import ProcessedReport
from agents.config import STORAGE_CONFIG
from src.utils.logger import get_logger


class ReportStorage:
    """Storage manager for processed threat reports with run-based naming"""
    
    def __init__(self, storage_dir: Optional[Path] = None, run_number: Optional[int] = None):
        """
        Initialize storage manager.
        
        Args:
            storage_dir: Directory for storing reports. If None, uses config default.
            run_number: Run number to tie reports to log file (e.g., 1 for agents_001.log)
        """
        self.logger = get_logger(__name__)
        self.storage_dir = Path(storage_dir) if storage_dir else STORAGE_CONFIG["storage_dir"]
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.run_number = run_number
        self.report_counter = 0  # Counter for reports in this run
        self.logger.info(f"Initialized storage at: {self.storage_dir}, run_number: {run_number}")
    
    def save_report(self, report: ProcessedReport) -> str:
        """
        Save processed report to storage with run-based naming.
        
        Args:
            report: Processed report to save
            
        Returns:
            File path where report was saved
        """
        # Generate filename based on run number
        if self.run_number is not None:
            # Increment report counter for this run
            self.report_counter += 1
            filename = f"agents_{self.run_number:03d}_report_{self.report_counter:03d}.json"
        else:
            # Fallback to old naming if run_number not set
            filename = f"{report.report_id}.json"
        
        filepath = self.storage_dir / filename
        
        # Convert to dictionary and add run metadata
        report_dict = report.to_dict()
        if self.run_number is not None:
            report_dict["run_number"] = self.run_number
            report_dict["report_number_in_run"] = self.report_counter
            report_dict["log_file"] = f"agents_{self.run_number:03d}.log"
        
        # Save as JSON
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved report {report.report_id} to {filepath} (run {self.run_number}, report {self.report_counter})")
            return str(filepath)
        
        except Exception as e:
            self.logger.error(f"Error saving report {report.report_id}: {e}")
            raise
    
    def load_report(self, report_id: str) -> Optional[Dict]:
        """
        Load a report by ID (searches both old and new naming schemes).
        
        Args:
            report_id: Report ID
            
        Returns:
            Report dictionary or None if not found
        """
        # Try new naming scheme first (run-based)
        pattern = f"agents_*_report_*.json"
        json_files = list(self.storage_dir.glob(pattern))
        
        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    report_dict = json.load(f)
                    if report_dict.get("report_id") == report_id:
                        return report_dict
            except Exception:
                continue
        
        # Try old naming scheme (fallback)
        filename = f"{report_id}.json"
        filepath = self.storage_dir / filename
        
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    report_dict = json.load(f)
                    self.logger.debug(f"Loaded report {report_id} (old naming scheme)")
                    return report_dict
            except Exception as e:
                self.logger.error(f"Error loading report {report_id}: {e}")
                return None
        
        self.logger.warning(f"Report {report_id} not found")
        return None
    
    def list_reports(
        self,
        limit: Optional[int] = None,
        sort_by: str = "processed_at",
        reverse: bool = True,
        run_number: Optional[int] = None
    ) -> List[Dict]:
        """
        List all stored reports.
        
        Args:
            limit: Maximum number of reports to return
            sort_by: Field to sort by (default: "processed_at")
            reverse: Sort in reverse order (newest first)
            run_number: Optional run number to filter by
            
        Returns:
            List of report dictionaries
        """
        reports = []
        
        # Find all JSON files
        if run_number is not None:
            # Filter by run number
            pattern = f"agents_{run_number:03d}_report_*.json"
            json_files = list(self.storage_dir.glob(pattern))
        else:
            # Get all JSON files (both old and new naming)
            json_files = list(self.storage_dir.glob("*.json"))
        
        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    report_dict = json.load(f)
                    reports.append(report_dict)
            except Exception as e:
                self.logger.warning(f"Error reading {filepath}: {e}")
                continue
        
        # Sort reports
        try:
            reports.sort(
                key=lambda x: x.get(sort_by, ""),
                reverse=reverse
            )
        except Exception as e:
            self.logger.warning(f"Error sorting reports: {e}")
        
        # Apply limit
        if limit:
            reports = reports[:limit]
        
        filter_msg = f" for run {run_number}" if run_number is not None else ""
        self.logger.info(f"Listed {len(reports)} reports{filter_msg}")
        return reports
    
    def get_report_count(self) -> int:
        """
        Get total number of stored reports.
        
        Returns:
            Number of reports
        """
        json_files = list(self.storage_dir.glob("*.json"))
        return len(json_files)
    
    def generate_report_id(self) -> str:
        """
        Generate a unique report ID.
        
        Returns:
            Unique report ID string
        """
        # Include run number in report ID if available
        if self.run_number is not None:
            return f"RPT-{self.run_number:03d}-{uuid.uuid4().hex[:8].upper()}"
        else:
            return f"RPT-{uuid.uuid4().hex[:12].upper()}"
    
    def get_reports_for_run(self, run_number: int) -> List[Dict]:
        """
        Get all reports for a specific run.
        
        Args:
            run_number: Run number to get reports for
            
        Returns:
            List of report dictionaries for the specified run
        """
        pattern = f"agents_{run_number:03d}_report_*.json"
        json_files = list(self.storage_dir.glob(pattern))
        
        reports = []
        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    report_dict = json.load(f)
                    reports.append(report_dict)
            except Exception as e:
                self.logger.warning(f"Error reading {filepath}: {e}")
                continue
        
        # Sort by report number in run
        reports.sort(key=lambda x: x.get("report_number_in_run", 0))
        
        self.logger.info(f"Found {len(reports)} reports for run {run_number}")
        return reports
    
    def delete_report(self, report_id: str) -> bool:
        """
        Delete a report by ID.
        
        Args:
            report_id: Report ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        filename = f"{report_id}.json"
        filepath = self.storage_dir / filename
        
        if not filepath.exists():
            self.logger.warning(f"Report {report_id} not found for deletion")
            return False
        
        try:
            filepath.unlink()
            self.logger.info(f"Deleted report {report_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting report {report_id}: {e}")
            return False

