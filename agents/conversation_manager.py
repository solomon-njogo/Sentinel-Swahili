"""
Conversation session manager for tracking active incomplete reports.
Manages sessions by phone number, handles timeouts, and persists to file.
"""

import json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta
from agents.config import STORAGE_CONFIG
from src.utils.logger import get_logger


class ConversationManager:
    """Manages active conversation sessions for incomplete reports"""
    
    def __init__(self, sessions_file: Optional[Path] = None):
        """
        Initialize conversation manager.
        
        Args:
            sessions_file: Path to sessions JSON file. If None, uses default.
        """
        self.logger = get_logger(__name__)
        self.storage_dir = Path(STORAGE_CONFIG["storage_dir"])
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Default sessions file location
        if sessions_file is None:
            self.sessions_file = self.storage_dir / "conversations.json"
        else:
            self.sessions_file = Path(sessions_file)
        
        # In-memory sessions: {phone_number: {report_id, last_updated, status}}
        self.sessions: Dict[str, Dict] = {}
        
        # Session timeout: 1 hour
        self.timeout_seconds = 3600
        
        # Load existing sessions from file
        self.load_sessions()
        self.logger.info(f"Initialized ConversationManager with {len(self.sessions)} active sessions")
    
    def load_sessions(self) -> None:
        """Load sessions from JSON file."""
        if not self.sessions_file.exists():
            self.logger.debug(f"Sessions file not found: {self.sessions_file}. Starting with empty sessions.")
            return
        
        try:
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.sessions = data
            
            # Clean up expired sessions on load
            self.cleanup_expired_sessions()
            
            self.logger.info(f"Loaded {len(self.sessions)} sessions from {self.sessions_file}")
        except Exception as e:
            self.logger.error(f"Error loading sessions from {self.sessions_file}: {e}")
            self.sessions = {}
    
    def save_sessions(self) -> None:
        """Save sessions to JSON file."""
        try:
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(self.sessions, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved {len(self.sessions)} sessions to {self.sessions_file}")
        except Exception as e:
            self.logger.error(f"Error saving sessions to {self.sessions_file}: {e}")
    
    def cleanup_expired_sessions(self) -> None:
        """Remove sessions that have exceeded the timeout."""
        now = datetime.now()
        expired_phones = []
        
        for phone_number, session in self.sessions.items():
            last_updated_str = session.get("last_updated")
            if not last_updated_str:
                # Invalid session, remove it
                expired_phones.append(phone_number)
                continue
            
            try:
                last_updated = datetime.fromisoformat(last_updated_str)
                elapsed = (now - last_updated).total_seconds()
                
                if elapsed > self.timeout_seconds:
                    expired_phones.append(phone_number)
                    self.logger.debug(f"Session expired for {phone_number} (elapsed: {elapsed:.0f}s)")
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Invalid timestamp in session for {phone_number}: {e}")
                expired_phones.append(phone_number)
        
        # Remove expired sessions
        for phone_number in expired_phones:
            del self.sessions[phone_number]
            self.logger.info(f"Removed expired session for {phone_number}")
        
        if expired_phones:
            self.save_sessions()
    
    def get_active_session(self, phone_number: str) -> Optional[Dict]:
        """
        Get active session for a phone number.
        
        Args:
            phone_number: Phone number (e.g., "whatsapp:+1234567890")
            
        Returns:
            Session dict with report_id, last_updated, status, or None if no active session
        """
        # Clean up expired sessions first
        self.cleanup_expired_sessions()
        
        session = self.sessions.get(phone_number)
        if not session:
            return None
        
        # Check if session is still valid (not expired)
        last_updated_str = session.get("last_updated")
        if not last_updated_str:
            return None
        
        try:
            last_updated = datetime.fromisoformat(last_updated_str)
            elapsed = (datetime.now() - last_updated).total_seconds()
            
            if elapsed > self.timeout_seconds:
                # Session expired, remove it
                del self.sessions[phone_number]
                self.save_sessions()
                self.logger.debug(f"Session expired for {phone_number}")
                return None
            
            return session
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid timestamp in session for {phone_number}: {e}")
            del self.sessions[phone_number]
            self.save_sessions()
            return None
    
    def create_session(self, phone_number: str, report_id: str) -> None:
        """
        Create a new session for a phone number.
        
        Args:
            phone_number: Phone number
            report_id: Report ID for this session
        """
        now = datetime.now().isoformat()
        self.sessions[phone_number] = {
            "report_id": report_id,
            "last_updated": now,
            "status": "incomplete"
        }
        self.save_sessions()
        self.logger.info(f"Created session for {phone_number} with report {report_id}")
    
    def update_session(self, phone_number: str, report_id: Optional[str] = None) -> None:
        """
        Update session timestamp (and optionally report_id).
        
        Args:
            phone_number: Phone number
            report_id: Optional new report ID (if updating to different report)
        """
        if phone_number not in self.sessions:
            if report_id:
                self.create_session(phone_number, report_id)
            return
        
        now = datetime.now().isoformat()
        self.sessions[phone_number]["last_updated"] = now
        
        if report_id:
            self.sessions[phone_number]["report_id"] = report_id
        
        self.save_sessions()
        self.logger.debug(f"Updated session for {phone_number}")
    
    def close_session(self, phone_number: str) -> None:
        """
        Close a session (mark as complete and remove).
        
        Args:
            phone_number: Phone number
        """
        if phone_number in self.sessions:
            report_id = self.sessions[phone_number].get("report_id", "unknown")
            del self.sessions[phone_number]
            self.save_sessions()
            self.logger.info(f"Closed session for {phone_number} (report {report_id} completed)")
    
    def has_active_session(self, phone_number: str) -> bool:
        """
        Check if phone number has an active session.
        
        Args:
            phone_number: Phone number
            
        Returns:
            True if active session exists, False otherwise
        """
        return self.get_active_session(phone_number) is not None

