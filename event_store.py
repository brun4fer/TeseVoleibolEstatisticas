"""
event_store.py
--------------
Persistent structured event storage for detected volleyball events.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2


CATEGORY_SPIKE = "spike"
CATEGORY_BLOCK = "block"
CATEGORY_ACE = "ace"
CATEGORY_ERROR = "error"            # ponto perdido por erro (atacante manda fora, na rede, etc.)
CATEGORY_FREEBALL = "freeball"
CATEGORY_BALL_ON_NET = "ball_on_net"
CATEGORY_UNDEFINED = "undefined"

ALL_CATEGORIES = (
    CATEGORY_SPIKE,
    CATEGORY_BLOCK,
    CATEGORY_ACE,
    CATEGORY_ERROR,
    CATEGORY_FREEBALL,
    CATEGORY_BALL_ON_NET,
    CATEGORY_UNDEFINED,
)


@dataclass
class StoredEvent:
    id: int
    type: str
    category: str
    point_type: str
    start_frame: Optional[int]
    end_frame: Optional[int]
    start_time_seconds: Optional[float]
    end_time_seconds: Optional[float]
    timestamp_label: str
    rally_duration_seconds: Optional[float]
    ball_avg_speed: Optional[float]
    ball_max_speed: Optional[float]
    point_team: Optional[str]
    attack_side: Optional[str]
    attack_team: Optional[str]
    defending_side: Optional[str]
    blocking_side: Optional[str]
    blocking_team: Optional[str]
    return_side: Optional[str]
    confidence: Optional[float]
    reason: str
    notes: str
    representative_frame_path: Optional[str] = None
    source_video_path: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class EventStore:
    def __init__(
        self,
        store_path: Path,
        preview_dir: Path,
        source_video_path: Optional[Path] = None,
        reset_on_start: bool = False,
        preview_max_width: int = 420,
    ) -> None:
        self.store_path = Path(store_path)
        self.preview_dir = Path(preview_dir)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.preview_max_width = max(120, int(preview_max_width))
        self.source_video_path = str(Path(source_video_path).resolve()) if source_video_path is not None else None
        if reset_on_start:
            self.data = self._new_data()
            self._save()
        else:
            self.data = self._load_or_create()
            if self.source_video_path is not None:
                self.data["source_video_path"] = self.source_video_path
                self._save()

    def reset(self) -> None:
        self.data = self._new_data()
        self._save()

    def record_event(self, event: Dict, preview_frame=None) -> Dict:
        event_id = int(self.data.get("next_id", 1))
        record = dict(event)
        record["id"] = event_id
        record.setdefault("source_video_path", self.data.get("source_video_path"))
        record.setdefault("timestamp_label", self._seconds_to_label(record.get("end_time_seconds")))
        preview_path = None
        if preview_frame is not None:
            preview_path = self._save_preview(event_id, preview_frame)
        record["representative_frame_path"] = preview_path

        stored = StoredEvent(
            id=int(record["id"]),
            type=str(record.get("type", CATEGORY_UNDEFINED)),
            category=str(record.get("category", CATEGORY_UNDEFINED)),
            point_type=str(record.get("point_type", "RALLY_ONLY")),
            start_frame=self._int_or_none(record.get("start_frame")),
            end_frame=self._int_or_none(record.get("end_frame")),
            start_time_seconds=self._float_or_none(record.get("start_time_seconds")),
            end_time_seconds=self._float_or_none(record.get("end_time_seconds")),
            timestamp_label=str(record.get("timestamp_label") or self._seconds_to_label(record.get("end_time_seconds"))),
            rally_duration_seconds=self._float_or_none(record.get("rally_duration_seconds")),
            ball_avg_speed=self._float_or_none(record.get("ball_avg_speed")),
            ball_max_speed=self._float_or_none(record.get("ball_max_speed")),
            point_team=record.get("point_team"),
            attack_side=record.get("attack_side"),
            attack_team=record.get("attack_team"),
            defending_side=record.get("defending_side"),
            blocking_side=record.get("blocking_side"),
            blocking_team=record.get("blocking_team"),
            return_side=record.get("return_side"),
            confidence=self._float_or_none(record.get("confidence")),
            reason=str(record.get("reason") or ""),
            notes=str(record.get("notes") or ""),
            representative_frame_path=record.get("representative_frame_path"),
            source_video_path=record.get("source_video_path"),
        )

        self.data["events"].append(stored.to_dict())
        self.data["next_id"] = event_id + 1
        self.data["updated_at"] = self._now()
        self._save()
        return stored.to_dict()

    def list_events(self, category: Optional[str] = None) -> List[Dict]:
        events = [self._normalize_loaded_event(event) for event in self.data.get("events", [])]
        if category is None:
            return events
        return [event for event in events if str(event.get("category")) == str(category)]

    def get_event(self, event_id: int) -> Optional[Dict]:
        for event in self.data.get("events", []):
            if int(event.get("id", -1)) == int(event_id):
                return self._normalize_loaded_event(event)
        return None

    def latest_event(self) -> Optional[Dict]:
        events = self.data.get("events", [])
        if not events:
            return None
        return dict(events[-1])

    def category_counts(self) -> Dict[str, int]:
        counts = {cat: 0 for cat in ALL_CATEGORIES}
        for event in self.data.get("events", []):
            category = str(event.get("category", CATEGORY_UNDEFINED))
            counts[category] = counts.get(category, 0) + 1
        return counts

    @classmethod
    def load_snapshot(cls, store_path: Path) -> Dict:
        path = Path(store_path)
        if not path.exists():
            return {
                "version": 1,
                "generated_at": None,
                "updated_at": None,
                "source_video_path": None,
                "next_id": 1,
                "events": [],
            }
        with open(path, "r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
        normalized_events = [cls._normalize_loaded_event(event) for event in snapshot.get("events", [])]
        snapshot["events"] = normalized_events
        return snapshot

    def _new_data(self) -> Dict:
        return {
            "version": 1,
            "generated_at": self._now(),
            "updated_at": self._now(),
            "source_video_path": self.source_video_path,
            "next_id": 1,
            "events": [],
        }

    def _load_or_create(self) -> Dict:
        if not self.store_path.exists():
            data = self._new_data()
            self._atomic_write(data)
            return data
        with open(self.store_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save(self) -> None:
        self._atomic_write(self.data)

    def _atomic_write(self, payload: Dict) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        tmp_path.replace(self.store_path)

    def _save_preview(self, event_id: int, frame) -> Optional[str]:
        if frame is None:
            return None
        image = frame.copy()
        height, width = image.shape[:2]
        if width > self.preview_max_width:
            scale = self.preview_max_width / float(width)
            image = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        preview_path = self.preview_dir / f"event_{event_id:04d}.png"
        ok = cv2.imwrite(str(preview_path), image)
        if not ok:
            return None
        return str(preview_path.resolve())

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _float_or_none(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int_or_none(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _seconds_to_label(value) -> str:
        seconds = EventStore._float_or_none(value)
        if seconds is None:
            return "--:--:--"
        total_seconds = max(0, int(round(seconds)))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _normalize_loaded_event(event: Dict) -> Dict:
        normalized = dict(event)
        normalized["timestamp_label"] = str(
            normalized.get("timestamp_label") or EventStore._seconds_to_label(normalized.get("end_time_seconds"))
        )
        normalized.setdefault("attack_team", None)
        normalized.setdefault("blocking_side", normalized.get("defending_side"))
        normalized.setdefault("blocking_team", None)
        normalized.setdefault("return_side", None)
        return normalized
