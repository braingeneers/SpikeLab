"""
Session management for storing SpikeData objects.

Manages in-memory storage of SpikeData objects keyed by session ID,
supporting multiple concurrent sessions.
"""

import time
import uuid
from typing import Dict, Optional

from spikedata import SpikeData


class SessionManager:
    """
    Manages sessions containing SpikeData objects.

    Sessions are keyed by session ID and can be retrieved, updated, or deleted.
    Supports automatic expiration of old sessions.
    """

    def __init__(self, default_ttl_seconds: int = 3600):
        """
        Initialize the session manager.

        Args:
            default_ttl_seconds: Default time-to-live for sessions in seconds (default: 1 hour)
        """
        self._sessions: Dict[str, Dict] = {}
        self.default_ttl = default_ttl_seconds

    def create_session(
        self, spikedata: SpikeData, ttl_seconds: Optional[int] = None
    ) -> str:
        """
        Create a new session with a SpikeData object.

        Args:
            spikedata: The SpikeData object to store
            ttl_seconds: Optional time-to-live for this session. If None, uses default.

        Returns:
            Session ID string
        """
        session_id = str(uuid.uuid4())
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        self._sessions[session_id] = {
            "spikedata": spikedata,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[SpikeData]:
        """
        Get a SpikeData object from a session.

        Args:
            session_id: The session ID

        Returns:
            SpikeData object if session exists and is not expired, None otherwise
        """
        if session_id not in self._sessions:
            return None

        session = self._sessions[session_id]
        if time.time() > session["expires_at"]:
            # Session expired, remove it
            del self._sessions[session_id]
            return None

        return session["spikedata"]

    def update_session(self, session_id: str, spikedata: SpikeData) -> bool:
        """
        Update an existing session with a new SpikeData object.

        Args:
            session_id: The session ID
            spikedata: The new SpikeData object

        Returns:
            True if session was updated, False if session doesn't exist
        """
        if session_id not in self._sessions:
            return False

        if time.time() > self._sessions[session_id]["expires_at"]:
            # Session expired, remove it
            del self._sessions[session_id]
            return False

        self._sessions[session_id]["spikedata"] = spikedata
        return True

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: The session ID

        Returns:
            True if session was deleted, False if it didn't exist
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def cleanup_expired(self) -> int:
        """
        Remove all expired sessions.

        Returns:
            Number of sessions removed
        """
        now = time.time()
        expired = [
            sid
            for sid, session in self._sessions.items()
            if now > session["expires_at"]
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def list_sessions(self) -> list:
        """
        List all active session IDs.

        Returns:
            List of active session IDs
        """
        self.cleanup_expired()
        return list(self._sessions.keys())

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get information about a session.

        Args:
            session_id: The session ID

        Returns:
            Dictionary with session info (spikedata, created_at, expires_at) or None
        """
        if session_id not in self._sessions:
            return None

        session = self._sessions[session_id]
        if time.time() > session["expires_at"]:
            del self._sessions[session_id]
            return None

        spikedata = session["spikedata"]
        return {
            "session_id": session_id,
            "num_neurons": spikedata.N,
            "length_ms": spikedata.length,
            "created_at": session["created_at"],
            "expires_at": session["expires_at"],
            "metadata": spikedata.metadata,
        }


# Global session manager instance
_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    return _session_manager
