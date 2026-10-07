"""
EYEON - Phase 4: Person-Object Association & Intelligence Module

Provides comprehensive spatial, motion, and temporal ownership intelligence:
1. Spatial association: IoU overlap, Euclidean center distance, body region heuristic
2. Motion consistency: Velocity vector direction & speed similarity (carried vs stationary)
3. Temporal ownership voting: 15-frame rolling window with 70% threshold (eliminates false switches)
4. Object lifecycle state machine:
   - UNKNOWN: newly detected, candidate owner voting in progress
   - CARRIED: confirmed owner carrying the object
   - DETACHED: owner separated/set object down, but nearby/within timeout
   - UNATTENDED: owner separated for > owner_timeout_seconds (5.0s alert trigger)
   - TRANSFERRED: handover/theft to a new confirmed person
"""

import sys
import math
import time
from pathlib import Path
from enum import Enum
from collections import deque
from typing import List, Dict, Any, Optional, Tuple, Set

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config_loader import load_config, get_config_value


class ObjectState(Enum):
    """Lifecycle states of a tracked object."""
    UNKNOWN = "UNKNOWN"
    CARRIED = "CARRIED"
    DETACHED = "DETACHED"
    UNATTENDED = "UNATTENDED"
    TRANSFERRED = "TRANSFERRED"


class PersonObjectAssociator:
    """
    Computes multi-factor spatial and motion association scores between
    people and everyday objects. Configurable via config.yaml.
    """

    def __init__(
        self,
        overlap_weight: Optional[float] = None,
        distance_weight: Optional[float] = None,
        body_region_weight: Optional[float] = None,
        motion_weight: Optional[float] = None,
        min_score: Optional[float] = None,
    ):
        config = load_config() if load_config else {}
        assoc_cfg = config.get("association", {})
        weights = assoc_cfg.get("weights", {})

        self.overlap_weight = overlap_weight if overlap_weight is not None else weights.get("overlap", 0.4)
        self.distance_weight = distance_weight if distance_weight is not None else weights.get("distance", 0.3)
        self.body_region_weight = body_region_weight if body_region_weight is not None else weights.get("body_region", 0.2)
        self.motion_weight = motion_weight if motion_weight is not None else weights.get("motion", 0.1)

        self.min_score = min_score if min_score is not None else assoc_cfg.get("min_score", 0.35)

    @staticmethod
    def calculate_iou(box_a: List[int], box_b: List[int]) -> float:
        """Calculate Intersection over Union (IoU)."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - intersection

        return (intersection / union) if union > 0 else 0.0

    @staticmethod
    def calculate_center_distance(person_box: List[int], object_box: List[int]) -> float:
        """Calculate normalized inverse distance between centers (1.0 = identical center)."""
        px1, py1, px2, py2 = person_box
        ox1, oy1, ox2, oy2 = object_box

        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0

        dist = math.sqrt((pcx - ocx) ** 2 + (pcy - ocy) ** 2)
        person_diag = math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)

        if person_diag <= 0:
            return 0.0

        norm_dist = dist / person_diag
        return max(0.0, min(1.0, 1.0 - norm_dist))

    @staticmethod
    def calculate_body_region_score(person_box: List[int], object_box: List[int]) -> float:
        """Checks if object center is within the person's expanded body region."""
        px1, py1, px2, py2 = person_box
        ox1, oy1, ox2, oy2 = object_box

        ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
        pw = px2 - px1
        ph = py2 - py1

        if pw <= 0 or ph <= 0:
            return 0.0

        exp_x1 = px1 - (pw * 0.20)
        exp_x2 = px2 + (pw * 0.20)
        exp_y1 = py1 - (ph * 0.10)
        exp_y2 = py2 + (ph * 0.10)

        return 1.0 if (exp_x1 <= ocx <= exp_x2 and exp_y1 <= ocy <= exp_y2) else 0.0

    @staticmethod
    def calculate_motion_score(
        person_history: Optional[List[Tuple[int, int]]] = None,
        object_history: Optional[List[Tuple[int, int]]] = None,
    ) -> float:
        """
        Calculates motion vector consistency between person and object.
        Score = 1.0 if moving in same direction or both stationary.
        Score = 0.0 if object is stationary on ground while person walks away.
        """
        if not person_history or not object_history or len(person_history) < 2 or len(object_history) < 2:
            return 0.5  # Neutral prior if insufficient history

        # Velocity vectors over last 2-3 positions
        p_curr, p_prev = person_history[-1], person_history[-2]
        o_curr, o_prev = object_history[-1], object_history[-2]

        vpx, vpy = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
        vox, voy = o_curr[0] - o_prev[0], o_curr[1] - o_prev[1]

        sp = math.sqrt(vpx ** 2 + vpy ** 2)
        so = math.sqrt(vox ** 2 + voy ** 2)

        # Case 1: Both stationary (person standing/sitting with object)
        if sp < 2.0 and so < 2.0:
            return 1.0

        # Case 2: One stationary, other moving (person walking away from static bag)
        if (sp >= 2.0 and so < 1.0) or (sp < 1.0 and so >= 2.0):
            return 0.0

        # Case 3: Both moving -> cosine similarity of direction vectors
        dot = (vpx * vox) + (vpy * voy)
        cos_sim = dot / (sp * so)
        return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))

    def calculate_score(
        self,
        person_box: List[int],
        object_box: List[int],
        person_history: Optional[List[Tuple[int, int]]] = None,
        object_history: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, float]:
        """Calculates weighted multi-factor association score."""
        overlap = self.calculate_iou(person_box, object_box)
        distance = self.calculate_center_distance(person_box, object_box)
        body = self.calculate_body_region_score(person_box, object_box)
        motion = self.calculate_motion_score(person_history, object_history)

        total_weight = (
            self.overlap_weight +
            self.distance_weight +
            self.body_region_weight +
            self.motion_weight
        )

        weighted_score = (
            (overlap * self.overlap_weight) +
            (distance * self.distance_weight) +
            (body * self.body_region_weight) +
            (motion * self.motion_weight)
        ) / (total_weight if total_weight > 0 else 1.0)

        return {
            "score": round(weighted_score, 4),
            "overlap": round(overlap, 4),
            "distance": round(distance, 4),
            "body_region": round(body, 4),
            "motion": round(motion, 4),
        }

    def associate(
        self,
        tracks: List[Dict[str, Any]],
        track_histories: Optional[Dict[int, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Associates objects to the closest matching person for the current frame.
        """
        persons = [t for t in tracks if t["class"] == "person"]
        objects = [t for t in tracks if t["class"] != "person"]

        associations = []
        track_histories = track_histories or {}

        for obj in objects:
            oid = obj["track_id"]
            o_hist = list(track_histories.get(oid, []))

            best_person = None
            best_res = None
            best_score = 0.0

            for p in persons:
                pid = p["track_id"]
                p_hist = list(track_histories.get(pid, []))

                res = self.calculate_score(
                    p["bbox"],
                    obj["bbox"],
                    person_history=p_hist,
                    object_history=o_hist,
                )

                if res["score"] > best_score:
                    best_score = res["score"]
                    best_person = p
                    best_res = res

            if best_person is not None and best_score >= self.min_score:
                associations.append({
                    "object_track_id": oid,
                    "person_track_id": best_person["track_id"],
                    "object_class": obj["class"],
                    "score": best_res["score"],
                    "overlap": best_res["overlap"],
                    "distance": best_res["distance"],
                    "body_region": best_res["body_region"],
                    "motion": best_res["motion"],
                })

        return associations


class TemporalOwnershipTracker:
    """
    Maintains a rolling window of frame-level associations and calculates
    temporal ownership voting to prevent accidental swaps and flicker.
    """

    def __init__(
        self,
        voting_window: Optional[int] = None,
        voting_ratio: Optional[float] = None,
    ):
        config = load_config() if load_config else {}
        assoc_cfg = config.get("association", {})

        self.voting_window = voting_window or assoc_cfg.get("voting_window", 15)
        self.voting_ratio = voting_ratio or assoc_cfg.get("voting_ratio", 0.70)

        # {object_track_id: deque([person_id_or_none, ...], maxlen=voting_window)}
        self.voting_history: Dict[int, deque] = {}
        # {object_track_id: confirmed_person_id}
        self.confirmed_owners: Dict[int, int] = {}

    def update(
        self,
        active_object_ids: Set[int],
        frame_associations: List[Dict[str, Any]],
    ) -> Dict[int, Optional[int]]:
        """
        Updates voting buffers and returns confirmed owner mapping:
        {object_track_id: confirmed_owner_id or None}
        """
        # Map current matches
        current_matches = {a["object_track_id"]: a["person_track_id"] for a in frame_associations}

        for oid in active_object_ids:
            if oid not in self.voting_history:
                self.voting_history[oid] = deque(maxlen=self.voting_window)

            match_pid = current_matches.get(oid, None)
            self.voting_history[oid].append(match_pid)

            # Count votes in window
            votes: Dict[int, int] = {}
            total_votes = len(self.voting_history[oid])

            for pid in self.voting_history[oid]:
                if pid is not None:
                    votes[pid] = votes.get(pid, 0) + 1

            # Check if any person meets voting ratio threshold and minimum vote count
            min_required_votes = max(1, int(self.voting_window * self.voting_ratio))
            highest_votes = 0
            candidate_owner = None
            for pid, count in votes.items():
                if count > highest_votes:
                    highest_votes = count
                    candidate_owner = pid

            ratio = (highest_votes / total_votes) if total_votes > 0 else 0.0
            if (
                candidate_owner is not None
                and highest_votes >= min_required_votes
                and ratio >= self.voting_ratio
            ):
                # Confirmed owner threshold met!
                self.confirmed_owners[oid] = candidate_owner

        # Cleanup disappeared objects
        stale = [oid for oid in self.voting_history if oid not in active_object_ids]
        for oid in stale:
            del self.voting_history[oid]
            self.confirmed_owners.pop(oid, None)

        return self.confirmed_owners


class ObjectStateManager:
    """
    Manages the lifecycle state machine for all tracked objects:
    CARRIED -> DETACHED -> UNATTENDED (Security Alert!)
    """

    def __init__(
        self,
        owner_timeout_seconds: Optional[float] = None,
        detachment_dist_thresh: float = 0.55,
    ):
        config = load_config() if load_config else {}
        self.owner_timeout_seconds = owner_timeout_seconds or config.get("association", {}).get("owner_timeout_seconds", 5.0)
        self.detachment_dist_thresh = detachment_dist_thresh

        # Object tracking states:
        # {oid: {
        #     "state": ObjectState,
        #     "confirmed_owner": int,
        #     "last_seen_near_owner": float,
        #     "unattended_since": Optional[float],
        #     "unattended_seconds": float
        # }}
        self.object_records: Dict[int, Dict[str, Any]] = {}

    def update_states(
        self,
        tracks: List[Dict[str, Any]],
        confirmed_owners: Dict[int, Optional[int]],
        frame_associations: List[Dict[str, Any]],
        current_time: float,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Updates and returns the state record for every active object.
        """
        person_tracks = {t["track_id"]: t for t in tracks if t["class"] == "person"}
        object_tracks = {t["track_id"]: t for t in tracks if t["class"] != "person"}
        current_assoc_map = {a["object_track_id"]: a for a in frame_associations}

        updated_records = {}

        for oid, obj in object_tracks.items():
            confirmed_owner = confirmed_owners.get(oid, None)

            if oid not in self.object_records:
                self.object_records[oid] = {
                    "state": ObjectState.UNKNOWN,
                    "confirmed_owner": confirmed_owner,
                    "last_seen_near_owner": current_time,
                    "unattended_since": None,
                    "unattended_seconds": 0.0,
                }

            rec = self.object_records[oid]
            rec["confirmed_owner"] = confirmed_owner

            if confirmed_owner is None:
                # No confirmed owner yet (still voting or abandoned from start)
                rec["state"] = ObjectState.UNKNOWN
                rec["unattended_seconds"] = 0.0
            else:
                # We have a confirmed owner! Check proximity
                owner_present = confirmed_owner in person_tracks

                if owner_present:
                    owner_box = person_tracks[confirmed_owner]["bbox"]
                    dist_score = PersonObjectAssociator.calculate_center_distance(owner_box, obj["bbox"])
                    is_near = (dist_score >= self.detachment_dist_thresh) or (oid in current_assoc_map and current_assoc_map[oid]["person_track_id"] == confirmed_owner)

                    if is_near:
                        # Owner is carrying or standing right next to the object
                        rec["state"] = ObjectState.CARRIED
                        rec["last_seen_near_owner"] = current_time
                        rec["unattended_since"] = None
                        rec["unattended_seconds"] = 0.0
                    else:
                        # Owner has separated / set it down and walked away
                        elapsed = current_time - rec["last_seen_near_owner"]
                        if elapsed >= self.owner_timeout_seconds:
                            rec["state"] = ObjectState.UNATTENDED
                            if rec["unattended_since"] is None:
                                rec["unattended_since"] = current_time
                            rec["unattended_seconds"] = round(elapsed, 1)
                        else:
                            rec["state"] = ObjectState.DETACHED
                            rec["unattended_seconds"] = round(elapsed, 1)
                else:
                    # Confirmed owner has walked completely out of frame!
                    elapsed = current_time - rec["last_seen_near_owner"]
                    if elapsed >= self.owner_timeout_seconds:
                        rec["state"] = ObjectState.UNATTENDED
                        if rec["unattended_since"] is None:
                            rec["unattended_since"] = current_time
                        rec["unattended_seconds"] = round(elapsed, 1)
                    else:
                        rec["state"] = ObjectState.DETACHED
                        rec["unattended_seconds"] = round(elapsed, 1)

            updated_records[oid] = rec

        # Cleanup removed tracks
        for oid in list(self.object_records.keys()):
            if oid not in object_tracks:
                del self.object_records[oid]

        return updated_records


class PersonObjectIntelligence:
    """
    Unified Phase 4 Engine combining Spatial Association,
    Temporal Voting, and Object Lifecycle State Management.
    """

    def __init__(self):
        self.associator = PersonObjectAssociator()
        self.voting_tracker = TemporalOwnershipTracker()
        self.state_manager = ObjectStateManager()

    def process(
        self,
        tracks: List[Dict[str, Any]],
        track_histories: Optional[Dict[int, Any]] = None,
        timestamp: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """
        Runs the complete Phase 4 pipeline.

        Returns:
            (frame_associations, object_states)
        """
        # 1. Multi-factor spatial & motion association
        associations = self.associator.associate(tracks, track_histories=track_histories)

        # 2. Temporal ownership voting
        active_obj_ids = {t["track_id"] for t in tracks if t["class"] != "person"}
        confirmed_owners = self.voting_tracker.update(active_obj_ids, associations)

        # 3. Object state management (CARRIED, DETACHED, UNATTENDED)
        object_states = self.state_manager.update_states(
            tracks=tracks,
            confirmed_owners=confirmed_owners,
            frame_associations=associations,
            current_time=timestamp,
        )

        return associations, object_states