from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    label: int
    confidence: float
    bbox: BBox


@dataclass
class SegmentDecision:
    should_save: bool
    segment_index: int
    reason: str


class SegmentTracker:
    """Tracks digit appearances so each visible segment is saved once."""

    def __init__(
        self,
        min_stable_frames: int = 2,
        missing_frames_to_close_segment: int = 5,
        min_frames_between_saves: int = 8,
        allow_same_label_motion_split: bool = False,
        same_label_center_distance: float = 0.45,
        label_change_center_distance: float = 0.18,
    ) -> None:
        self.min_stable_frames = max(1, int(min_stable_frames))
        self.missing_frames_to_close_segment = max(1, int(missing_frames_to_close_segment))
        self.min_frames_between_saves = max(1, int(min_frames_between_saves))
        self.allow_same_label_motion_split = bool(allow_same_label_motion_split)
        self.same_label_center_distance = float(same_label_center_distance)
        self.label_change_center_distance = float(label_change_center_distance)

        self.active_label: Optional[int] = None
        self.active_bbox: Optional[BBox] = None
        self.candidate_label: Optional[int] = None
        self.candidate_bbox: Optional[BBox] = None
        self.candidate_count = 0
        self.missing_count = 0
        self.frames_since_save = 10**9
        self.segment_index = 0

    def update(
        self,
        detection: Optional[Detection],
        frame_width: int,
        frame_height: int,
    ) -> SegmentDecision:
        self.frames_since_save += 1

        if detection is None:
            self.missing_count += 1
            self.candidate_label = None
            self.candidate_bbox = None
            self.candidate_count = 0
            if self.missing_count >= self.missing_frames_to_close_segment:
                self.active_label = None
                self.active_bbox = None
            return SegmentDecision(False, self.segment_index, "missing")

        self.missing_count = 0

        if detection.label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = detection.label
            self.candidate_count = 1

        self.candidate_bbox = detection.bbox

        if self.candidate_count < self.min_stable_frames:
            return SegmentDecision(False, self.segment_index, "warming_up")

        if self.active_label is None:
            return self._save(detection, "new_segment")

        if detection.label != self.active_label:
            if self.frames_since_save < self.min_frames_between_saves:
                return SegmentDecision(False, self.segment_index, "save_cooldown")
            if (
                self.active_bbox is not None
                and self._center_distance(detection.bbox, self.active_bbox, frame_width, frame_height)
                < self.label_change_center_distance
            ):
                self.active_bbox = detection.bbox
                return SegmentDecision(False, self.segment_index, "label_jitter_same_object")
            return self._save(detection, "label_changed")

        if (
            self.allow_same_label_motion_split
            and self.active_bbox is not None
            and self.frames_since_save >= self.min_frames_between_saves
            and self._center_distance(detection.bbox, self.active_bbox, frame_width, frame_height)
            >= self.same_label_center_distance
        ):
            return self._save(detection, "same_label_moved")

        return SegmentDecision(False, self.segment_index, "same_segment")

    def _save(self, detection: Detection, reason: str) -> SegmentDecision:
        idx = self.segment_index
        self.segment_index += 1
        self.active_label = detection.label
        self.active_bbox = detection.bbox
        self.frames_since_save = 0
        return SegmentDecision(True, idx, reason)

    @staticmethod
    def _center_distance(a: BBox, b: BBox, frame_width: int, frame_height: int) -> float:
        ax = (a[0] + a[2]) * 0.5
        ay = (a[1] + a[3]) * 0.5
        bx = (b[0] + b[2]) * 0.5
        by = (b[1] + b[3]) * 0.5
        dx = (ax - bx) / max(1, frame_width)
        dy = (ay - by) / max(1, frame_height)
        return (dx * dx + dy * dy) ** 0.5
