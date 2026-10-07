"""
EYEON - Alert Management & State Machine Module
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import cv2
import numpy as np

from src.database.db import DatabaseManager
from src.risk.risk_engine import RiskRecord, RiskLevel
from src.utils.config_loader import load_config, get_path


class AlertState(Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


@dataclass
class Alert:
    """
    Representation of an active or historical security alert.
    """
    alert_id: str
    person_track_id: int
    risk_score: float
    risk_level: RiskLevel
    reasons: List[str]
    timestamp: float
    datetime_str: str
    associated_objects: List[int] = field(default_factory=list)
    zones: List[str] = field(default_factory=list)
    snapshot_path: Optional[str] = None
    state: AlertState = AlertState.ACTIVE


class AlertManager:
    """
    State machine and manager for security alerts, snapshot saving, and database integration.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        snapshot_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or load_config()
        db_cfg = cfg.get("database", {})

        if snapshot_dir is None:
            sn_path = db_cfg.get("snapshots", "data/alerts")
            self.snapshot_dir = Path(sn_path)
        else:
            self.snapshot_dir = Path(snapshot_dir)

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.db = db_manager or DatabaseManager()

        # In-memory alert state tracking
        # {person_track_id: Alert}
        self.active_alerts: Dict[int, Alert] = {}

        # Counter for sequential alert ID generation
        self._alert_counter = 0

    def process_risk_records(
        self,
        risk_records: Dict[int, RiskRecord],
        frame: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> List[Alert]:
        """
        Processes frame risk records, creates new alerts, saves snapshots,
        logs to database, and auto-resolves cleared alerts.

        Returns:
            List of newly fired Alert objects this frame.
        """
        newly_fired: List[Alert] = []
        current_pids = set(risk_records.keys())

        for pid, record in risk_records.items():
            # 1. Check if a new alert was fired by RiskEngine
            if record.alert_fired:
                alert_id = self._generate_alert_id(pid, timestamp)
                snapshot_file = None

                # Save snapshot image if frame provided
                if frame is not None:
                    snapshot_file = self._save_snapshot(alert_id, frame)

                alert = Alert(
                    alert_id=alert_id,
                    person_track_id=pid,
                    risk_score=record.risk_score,
                    risk_level=record.risk_level,
                    reasons=record.reasons,
                    timestamp=timestamp,
                    datetime_str=datetime.fromtimestamp(timestamp).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    associated_objects=record.associated_objects,
                    zones=record.zone_names,
                    snapshot_path=str(snapshot_file) if snapshot_file else None,
                    state=AlertState.ACTIVE,
                )

                # Track active alert in memory
                self.active_alerts[pid] = alert
                newly_fired.append(alert)

                # Persist to database
                self.db.log_event(
                    alert_id=alert.alert_id,
                    timestamp=alert.timestamp,
                    person_track_id=alert.person_track_id,
                    risk_score=alert.risk_score,
                    risk_level=alert.risk_level.value,
                    reasons=alert.reasons,
                    associated_objects=alert.associated_objects,
                    zones=alert.zones,
                    snapshot_path=alert.snapshot_path,
                    status=alert.state.value,
                )

            # 2. Update existing active alert if risk resolved or person left
            elif pid in self.active_alerts:
                active_alert = self.active_alerts[pid]
                if record.risk_level == RiskLevel.LOW:
                    # Risk cleared -> transition to RESOLVED
                    self._resolve_alert(pid, timestamp)

        # 3. Handle persons who disappeared from tracking while having active alerts
        for pid in list(self.active_alerts.keys()):
            if pid not in current_pids:
                self._resolve_alert(pid, timestamp)

        return newly_fired

    def update_alert_state(self, alert_id: str, new_state: AlertState) -> bool:
        """
        Manually updates the state of an alert (e.g. ACKNOWLEDGED or RESOLVED by an operator).
        """
        # Update in DB
        success = self.db.update_event_status(alert_id, new_state.value)

        # Update in active alerts if present
        for pid, alert in list(self.active_alerts.items()):
            if alert.alert_id == alert_id:
                alert.state = new_state
                if new_state == AlertState.RESOLVED:
                    del self.active_alerts[pid]
                break

        return success

    def get_active_alerts(self) -> List[Alert]:
        """Returns list of currently active alerts."""
        return list(self.active_alerts.values())

    # ------------------------------------------------------------------
    # Private Helper Methods
    # ------------------------------------------------------------------

    def _generate_alert_id(self, person_id: int, timestamp: float) -> str:
        self._alert_counter += 1
        ts_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        return f"ALT_{ts_str}_P{person_id}_{self._alert_counter:03d}"

    def _save_snapshot(self, alert_id: str, frame: np.ndarray) -> Path:
        filename = f"{alert_id}.jpg"
        filepath = self.snapshot_dir / filename
        try:
            cv2.imwrite(str(filepath), frame)
            return filepath
        except Exception as e:
            print(f"Failed to save snapshot {filepath}: {e}")
            return filepath

    def _resolve_alert(self, person_id: int, timestamp: float) -> None:
        if person_id in self.active_alerts:
            alert = self.active_alerts.pop(person_id)
            alert.state = AlertState.RESOLVED
            self.db.update_event_status(alert.alert_id, AlertState.RESOLVED.value)
