"""
EYEON - Phase 6: Explainable Risk Engine

Combines outputs from Phase 4 (Association + Object States)
and Phase 5 (Zone membership + Dwell time) to compute a
per-person risk score with an explainable reason string.

Components:
  RiskLevel       — LOW / MEDIUM / HIGH enum
  RiskRecord      — Full per-person risk result for one frame
  RiskEngine      — Stateful scorer: computes score, classifies level,
                    manages confirm-frames hysteresis and alert cooldown
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config_loader import load_config
from src.association.association import ObjectState
from src.positioning.zones import ZoneInfo


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


@dataclass
class RiskRecord:
    """
    Full risk assessment for one person in one frame.

    Attributes:
        person_track_id:       Tracker ID of the person.
        risk_score:            Raw composite score in [0.0, 1.0].
        risk_level:            Classified level: LOW / MEDIUM / HIGH.
        alert_fired:           True if a new alert was fired this frame.
        reasons:               Human-readable explanation list.
        component_scores:      Dict of individual factor scores.
        associated_objects:    List of object track IDs linked to this person.
        zone_names:            Zones the person currently occupies.
        dwell_seconds:         Dwell time per zone.
    """
    person_track_id:    int
    risk_score:         float
    risk_level:         RiskLevel
    alert_fired:        bool                = False
    reasons:            List[str]           = field(default_factory=list)
    component_scores:   Dict[str, float]    = field(default_factory=dict)
    associated_objects: List[int]           = field(default_factory=list)
    zone_names:         List[str]           = field(default_factory=list)
    dwell_seconds:      Dict[str, float]    = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    Stateful, explainable risk scorer for EYEON.

    Scoring factors (all configurable in config.yaml → risk.weights):
      1. object_confidence  — presence of a confirmed/unattended object
      2. zone               — risk weight of occupied zone(s)
      3. association_stability — how stable the person-object link is
      4. dwell_time         — how long the person has been in a risk zone

    Alert hysteresis:
      - A HIGH-risk score must persist for `confirm_frames` consecutive frames
        before an alert is fired, to suppress transient spikes.
      - Each person has an independent `alert_cooldown_seconds` timer so
        the same person does not spam alerts.
    """

    # Maximum dwell time used for normalisation (seconds)
    MAX_DWELL_NORM: float = 30.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg      = config or load_config()
        risk_cfg = cfg.get("risk", {})
        w        = risk_cfg.get("weights", {})

        # Factor weights
        self.w_object_conf  = float(w.get("object_confidence", 0.35))
        self.w_zone         = float(w.get("zone",              0.25))
        self.w_assoc_stab   = float(w.get("association_stability", 0.20))
        self.w_dwell        = float(w.get("dwell_time",        0.20))

        # Thresholds
        levels = risk_cfg.get("levels", {})
        self.thresh_medium = float(levels.get("medium", 0.40))
        self.thresh_high   = float(levels.get("high",   0.70))

        # Hysteresis
        self.confirm_frames         = int(risk_cfg.get("confirm_frames", 15))
        self.alert_cooldown_seconds = float(risk_cfg.get("alert_cooldown_seconds", 10.0))

        # Per-person state
        # {person_id: {"high_streak": int, "last_alert_time": float}}
        self._person_state: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        tracks:        List[Dict[str, Any]],
        associations:  List[Dict[str, Any]],
        object_states: Dict[int, Dict[str, Any]],
        zone_info:     Dict[int, ZoneInfo],
        timestamp:     float = 0.0,
    ) -> Dict[int, RiskRecord]:
        """
        Compute risk for every person in the current frame.

        Args:
            tracks:        Full track list (persons + objects).
            associations:  Phase 4 association list.
            object_states: Phase 4 object state records.
            zone_info:     Phase 5 zone info per track_id.
            timestamp:     Current video time in seconds.

        Returns:
            Dict mapping person_track_id → RiskRecord.
        """
        persons = [t for t in tracks if t["class"] == "person"]

        # Build helper lookups
        # person_id → list of associated object track IDs
        # Include both:
        #   (a) current-frame associations (object actively near person)
        #   (b) objects whose confirmed_owner is this person even when
        #       the owner has walked away (most important for UNATTENDED alerts)
        person_objects: Dict[int, List[int]] = {p["track_id"]: [] for p in persons}
        for assoc in associations:
            pid = assoc["person_track_id"]
            oid = assoc["object_track_id"]
            if pid in person_objects and oid not in person_objects[pid]:
                person_objects[pid].append(oid)
        for oid, state_rec in object_states.items():
            owner_pid = state_rec.get("confirmed_owner")
            if owner_pid in person_objects and oid not in person_objects[owner_pid]:
                person_objects[owner_pid].append(oid)

        results: Dict[int, RiskRecord] = {}

        # Expire state for persons no longer present
        active_pids = {p["track_id"] for p in persons}
        for pid in list(self._person_state.keys()):
            if pid not in active_pids:
                del self._person_state[pid]

        for person in persons:
            pid = person["track_id"]

            # Initialise per-person state tracker
            if pid not in self._person_state:
                self._person_state[pid] = {
                    "high_streak":      0,
                    "last_alert_time":  -999.0,
                }

            obj_ids  = person_objects.get(pid, [])
            zi       = zone_info.get(pid)

            # ----------------------------------------------------------------
            # Score component 1: object_confidence
            # ----------------------------------------------------------------
            obj_conf_score, obj_reasons = self._score_object_confidence(
                obj_ids, object_states
            )

            # ----------------------------------------------------------------
            # Score component 2: zone
            # ----------------------------------------------------------------
            zone_score, zone_reasons = self._score_zone(zi)

            # ----------------------------------------------------------------
            # Score component 3: association_stability
            # ----------------------------------------------------------------
            assoc_stab_score, assoc_reasons = self._score_association_stability(
                pid, associations, obj_ids, object_states
            )

            # ----------------------------------------------------------------
            # Score component 4: dwell_time
            # ----------------------------------------------------------------
            dwell_score, dwell_reasons = self._score_dwell(zi)

            # ----------------------------------------------------------------
            # Weighted composite score (sum of weights = 1.0)
            # ----------------------------------------------------------------
            total_w = (
                self.w_object_conf + self.w_zone
                + self.w_assoc_stab + self.w_dwell
            )
            raw_score = (
                obj_conf_score  * self.w_object_conf
                + zone_score    * self.w_zone
                + assoc_stab_score * self.w_assoc_stab
                + dwell_score   * self.w_dwell
            ) / (total_w if total_w > 0 else 1.0)

            risk_score = round(min(1.0, max(0.0, raw_score)), 4)

            # ----------------------------------------------------------------
            # Classify level
            # ----------------------------------------------------------------
            if risk_score >= self.thresh_high:
                level = RiskLevel.HIGH
            elif risk_score >= self.thresh_medium:
                level = RiskLevel.MEDIUM
            else:
                level = RiskLevel.LOW

            # ----------------------------------------------------------------
            # Hysteresis: require confirm_frames consecutive HIGH frames
            # ----------------------------------------------------------------
            ps = self._person_state[pid]
            if level == RiskLevel.HIGH:
                ps["high_streak"] += 1
            else:
                ps["high_streak"] = 0

            # ----------------------------------------------------------------
            # Alert cooldown
            # ----------------------------------------------------------------
            alert_fired = False
            if (
                level == RiskLevel.HIGH
                and ps["high_streak"] >= self.confirm_frames
                and (timestamp - ps["last_alert_time"]) >= self.alert_cooldown_seconds
            ):
                alert_fired = True
                ps["last_alert_time"] = timestamp

            # ----------------------------------------------------------------
            # Collect all reasons (only include relevant ones)
            # ----------------------------------------------------------------
            all_reasons = obj_reasons + zone_reasons + assoc_reasons + dwell_reasons
            if not all_reasons:
                all_reasons = ["No significant risk factors detected."]

            results[pid] = RiskRecord(
                person_track_id    = pid,
                risk_score         = risk_score,
                risk_level         = level,
                alert_fired        = alert_fired,
                reasons            = all_reasons,
                component_scores   = {
                    "object_confidence":     round(obj_conf_score, 3),
                    "zone":                  round(zone_score, 3),
                    "association_stability": round(assoc_stab_score, 3),
                    "dwell_time":            round(dwell_score, 3),
                },
                associated_objects = obj_ids,
                zone_names         = zi.current_zones if zi else [],
                dwell_seconds      = zi.dwell_seconds  if zi else {},
            )

        return results

    # ------------------------------------------------------------------
    # Scoring components (private)
    # ------------------------------------------------------------------

    def _score_object_confidence(
        self,
        obj_ids:       List[int],
        object_states: Dict[int, Dict[str, Any]],
    ) -> Tuple[float, List[str]]:
        """
        Score = 0   no associated objects
              = 0.3  objects in UNKNOWN state
              = 0.5  object CARRIED
              = 0.8  object DETACHED
              = 1.0  object UNATTENDED
        """
        if not obj_ids:
            return 0.0, []

        max_score = 0.0
        reasons   = []

        for oid in obj_ids:
            rec   = object_states.get(oid, {})
            state = rec.get("state", ObjectState.UNKNOWN)
            secs  = rec.get("unattended_seconds", 0.0)

            if state == ObjectState.UNATTENDED:
                s = 1.0
                reasons.append(f"Object #{oid} UNATTENDED for {secs:.0f}s")
            elif state == ObjectState.DETACHED:
                s = 0.8
                reasons.append(f"Object #{oid} DETACHED for {secs:.0f}s")
            elif state == ObjectState.CARRIED:
                s = 0.5
            else:
                s = 0.3

            max_score = max(max_score, s)

        return max_score, reasons

    def _score_zone(
        self,
        zi: Optional[ZoneInfo],
    ) -> Tuple[float, List[str]]:
        """
        Score = zone_weight of the highest-risk zone occupied.
        0.0 if outside all zones.
        """
        if zi is None or not zi.current_zones:
            return 0.0, []

        score   = zi.zone_weight
        reasons = [f"In zone '{z}' (weight={zi.zone_weight:.2f})" for z in zi.current_zones]
        return score, reasons

    def _score_association_stability(
        self,
        pid:           int,
        associations:  List[Dict[str, Any]],
        obj_ids:       List[int],
        object_states: Dict[int, Dict[str, Any]],
    ) -> Tuple[float, List[str]]:
        """
        Score = mean of raw association scores for this person's objects.
        Higher score = more stable link = more confident attribution.
        Unattended objects with no current frame link score 1.0 (most concerning).
        """
        if not obj_ids:
            return 0.0, []

        scores = []
        for oid in obj_ids:
            # Find association score this frame
            frame_assoc = next(
                (a for a in associations
                 if a["object_track_id"] == oid and a["person_track_id"] == pid),
                None,
            )
            if frame_assoc:
                scores.append(frame_assoc["score"])
            else:
                # Object is confirmed-owned but not in frame associations:
                # owner walked away → high concern
                rec   = object_states.get(oid, {})
                state = rec.get("state", ObjectState.UNKNOWN)
                if state in (ObjectState.UNATTENDED, ObjectState.DETACHED):
                    scores.append(1.0)
                else:
                    scores.append(0.0)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        return mean_score, []

    def _score_dwell(
        self,
        zi: Optional[ZoneInfo],
    ) -> Tuple[float, List[str]]:
        """
        Score = max dwell time in any zone, normalised to MAX_DWELL_NORM.
        Capped at 1.0.
        """
        if zi is None or not zi.dwell_seconds:
            return 0.0, []

        max_dwell = max(zi.dwell_seconds.values(), default=0.0)
        score     = min(1.0, max_dwell / self.MAX_DWELL_NORM)
        reasons   = []

        if max_dwell >= 10.0:
            zone = max(zi.dwell_seconds, key=zi.dwell_seconds.get)
            reasons.append(f"Dwell in '{zone}': {max_dwell:.0f}s")

        return score, reasons
