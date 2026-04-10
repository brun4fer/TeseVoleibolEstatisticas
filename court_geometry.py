"""
court_geometry.py
-----------------
Reusable geometric layer built from court calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


SIDE_A = "CampoA"
SIDE_B = "CampoB"
COURT_W_M = 9.0
COURT_L_M = 18.0


@dataclass(frozen=True)
class CourtZones:
    court: np.ndarray
    left: np.ndarray
    right: np.ndarray
    net: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "court": self.court.copy(),
            "left": self.left.copy(),
            "right": self.right.copy(),
            "net": self.net.copy(),
        }


class CourtGeometry:
    def __init__(
        self,
        H: np.ndarray,
        net_line: Tuple[Tuple[int, int], Tuple[int, int]],
        court_margin_m: float = 0.35,
        neutral_tolerance_px: float = 15.0,
        net_zone_tolerance_px: float = 25.0,
    ) -> None:
        self.H = np.asarray(H, dtype=np.float32)
        self.H_inv = np.linalg.inv(self.H)
        self.net_line = (
            (float(net_line[0][0]), float(net_line[0][1])),
            (float(net_line[1][0]), float(net_line[1][1])),
        )
        self.court_margin_m = float(court_margin_m)
        self.neutral_tolerance_px = float(neutral_tolerance_px)
        self.net_zone_tolerance_px = float(net_zone_tolerance_px)

        court_polygon = self._court_polygon_from_homography()
        left_polygon = self._half_court_polygon((0.0, 0.0), (COURT_W_M, COURT_L_M / 2.0))
        right_polygon = self._half_court_polygon((0.0, COURT_L_M / 2.0), (COURT_W_M, COURT_L_M))
        assigned_left, assigned_right = self._assign_half_courts(left_polygon, right_polygon)
        net_polygon = self._build_net_zone_polygon(self.net_zone_tolerance_px)

        self.zones = CourtZones(
            court=court_polygon,
            left=assigned_left,
            right=assigned_right,
            net=net_polygon,
        )
        self.court_top_y = float(np.min(self.zones.court[:, 1]))

    def pixel_to_court(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        p = np.array([[float(pt[0])], [float(pt[1])], [1.0]], dtype=np.float32)
        proj = self.H @ p
        if abs(float(proj[2])) < 1e-8:
            return 0.0, 0.0
        proj /= proj[2]
        return float(proj[0]), float(proj[1])

    def court_to_pixel(self, pt: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        p = np.array([[float(pt[0])], [float(pt[1])], [1.0]], dtype=np.float32)
        proj = self.H_inv @ p
        if abs(float(proj[2])) < 1e-8:
            return None
        proj /= proj[2]
        return float(proj[0]), float(proj[1])

    def is_inside_court(
        self,
        point: Tuple[float, float],
        point_space: str = "pixel",
        margin_m: Optional[float] = None,
    ) -> bool:
        margin = self.court_margin_m if margin_m is None else float(margin_m)
        if point_space == "court":
            x, y = float(point[0]), float(point[1])
        else:
            x, y = self.pixel_to_court(point)
        return (-margin) <= x <= (COURT_W_M + margin) and (-margin) <= y <= (COURT_L_M + margin)

    def is_out_of_bounds(
        self,
        point: Tuple[float, float],
        point_space: str = "pixel",
        margin_m: Optional[float] = None,
    ) -> bool:
        return not self.is_inside_court(point, point_space=point_space, margin_m=margin_m)

    def signed_distance_to_net(self, point: Tuple[float, float]) -> float:
        (x1, y1), (x2, y2) = self.net_line
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length <= 1e-9:
            return 0.0
        signed_area = dx * (float(point[1]) - y1) - dy * (float(point[0]) - x1)
        return float(signed_area / length)

    def get_side_of_net(
        self,
        point: Tuple[float, float],
        neutral_tolerance_px: Optional[float] = None,
    ) -> Optional[str]:
        tolerance = self.neutral_tolerance_px if neutral_tolerance_px is None else float(neutral_tolerance_px)
        signed = self.signed_distance_to_net(point)
        if abs(signed) <= tolerance:
            return None
        return SIDE_A if signed > 0.0 else SIDE_B

    def other_side(self, side: Optional[str]) -> Optional[str]:
        if side == SIDE_A:
            return SIDE_B
        if side == SIDE_B:
            return SIDE_A
        return None

    def project_point_to_net(self, point: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
        (x1, y1), (x2, y2) = self.net_line
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return (float(x1), float(y1)), 1e9
        t = ((float(point[0]) - x1) * dx + (float(point[1]) - y1) * dy) / denom
        t = max(0.0, min(1.0, float(t)))
        proj = (float(x1 + t * dx), float(y1 + t * dy))
        dist = float(np.hypot(float(point[0]) - proj[0], float(point[1]) - proj[1]))
        return proj, dist

    def distance_to_net(self, point: Tuple[float, float]) -> float:
        _proj, dist = self.project_point_to_net(point)
        return dist

    def is_in_net_zone(self, point: Tuple[float, float], tolerance_px: Optional[float] = None) -> bool:
        tolerance = self.net_zone_tolerance_px if tolerance_px is None else float(tolerance_px)
        return self.distance_to_net(point) <= tolerance

    def did_cross_net(
        self,
        prev_point: Tuple[float, float],
        curr_point: Tuple[float, float],
        neutral_tolerance_px: Optional[float] = None,
    ) -> bool:
        tolerance = self.neutral_tolerance_px if neutral_tolerance_px is None else float(neutral_tolerance_px)
        prev_signed = self.signed_distance_to_net(prev_point)
        curr_signed = self.signed_distance_to_net(curr_point)
        if abs(prev_signed) <= tolerance or abs(curr_signed) <= tolerance:
            return True
        return prev_signed * curr_signed < 0.0

    def segment_hits_net_zone(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        tolerance_px: Optional[float] = None,
        samples: int = 10,
    ) -> bool:
        tolerance = self.net_zone_tolerance_px if tolerance_px is None else float(tolerance_px)
        if self.is_in_net_zone(p0, tolerance) or self.is_in_net_zone(p1, tolerance):
            return True
        for idx in range(1, max(2, int(samples))):
            alpha = idx / float(max(2, int(samples)))
            x = float(p0[0] + (p1[0] - p0[0]) * alpha)
            y = float(p0[1] + (p1[1] - p0[1]) * alpha)
            if self.is_in_net_zone((x, y), tolerance):
                return True
        return False

    def net_relative_motion(
        self,
        prev_point: Tuple[float, float],
        curr_point: Tuple[float, float],
    ) -> Dict[str, float | bool]:
        prev_signed = self.signed_distance_to_net(prev_point)
        curr_signed = self.signed_distance_to_net(curr_point)
        prev_abs = abs(prev_signed)
        curr_abs = abs(curr_signed)
        delta_abs = float(prev_abs - curr_abs)
        return {
            "prev_signed_distance": float(prev_signed),
            "curr_signed_distance": float(curr_signed),
            "towards_net_delta": float(delta_abs),
            "away_from_net_delta": float(curr_abs - prev_abs),
            "moving_towards_net": bool(delta_abs > 0.0),
            "moving_away_from_net": bool(curr_abs > prev_abs),
        }

    def motion_towards_net(
        self,
        prev_point: Tuple[float, float],
        curr_point: Tuple[float, float],
        min_delta_px: float = 0.0,
    ) -> bool:
        delta = self.net_relative_motion(prev_point, curr_point)
        return bool(float(delta["towards_net_delta"]) > float(min_delta_px))

    def get_court_zones(self) -> Dict[str, np.ndarray]:
        return self.zones.as_dict()

    def point_zone(self, point: Tuple[float, float]) -> str:
        if self.is_out_of_bounds(point):
            return "out"
        if self.is_in_net_zone(point):
            return "net"
        side = self.get_side_of_net(point)
        if side == SIDE_A:
            return "left"
        if side == SIDE_B:
            return "right"
        return "net"

    def estimate_pixels_per_meter_near_net(self, meters: float) -> float:
        if meters <= 0.0:
            return 0.0
        net_center_px = (
            (self.net_line[0][0] + self.net_line[1][0]) / 2.0,
            (self.net_line[0][1] + self.net_line[1][1]) / 2.0,
        )
        net_center_court = self.pixel_to_court(net_center_px)

        px_per_meter_samples = []
        for dx, dy in ((meters, 0.0), (-meters, 0.0), (0.0, meters), (0.0, -meters)):
            court_pt = (net_center_court[0] + dx, net_center_court[1] + dy)
            if self.is_out_of_bounds(court_pt, point_space="court", margin_m=0.0):
                continue
            pixel_pt = self.court_to_pixel(court_pt)
            if pixel_pt is None:
                continue
            dist_px = float(np.hypot(pixel_pt[0] - net_center_px[0], pixel_pt[1] - net_center_px[1]))
            px_per_meter_samples.append(dist_px / meters)

        if px_per_meter_samples:
            return float(np.median(px_per_meter_samples) * meters)

        net_len_px = float(
            np.hypot(
                self.net_line[1][0] - self.net_line[0][0],
                self.net_line[1][1] - self.net_line[0][1],
            )
        )
        return max(net_len_px / COURT_W_M, 1.0) * meters

    def _court_polygon_from_homography(self) -> np.ndarray:
        points = []
        for court_pt in ((0.0, 0.0), (COURT_W_M, 0.0), (COURT_W_M, COURT_L_M), (0.0, COURT_L_M)):
            pixel_pt = self.court_to_pixel(court_pt)
            if pixel_pt is None:
                raise ValueError("Could not project court polygon from homography.")
            points.append((float(pixel_pt[0]), float(pixel_pt[1])))
        return np.array(points, dtype=np.float32)

    def _half_court_polygon(
        self,
        court_min: Tuple[float, float],
        court_max: Tuple[float, float],
    ) -> np.ndarray:
        x0, y0 = court_min
        x1, y1 = court_max
        points = []
        for court_pt in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            pixel_pt = self.court_to_pixel(court_pt)
            if pixel_pt is None:
                raise ValueError("Could not project half-court polygon from homography.")
            points.append((float(pixel_pt[0]), float(pixel_pt[1])))
        return np.array(points, dtype=np.float32)

    def _assign_half_courts(
        self,
        first_half: np.ndarray,
        second_half: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        first_center = self._polygon_center(first_half)
        second_center = self._polygon_center(second_half)
        first_side = self.get_side_of_net(first_center, neutral_tolerance_px=1.0)
        second_side = self.get_side_of_net(second_center, neutral_tolerance_px=1.0)

        if first_side == SIDE_A and second_side == SIDE_B:
            return first_half, second_half
        if first_side == SIDE_B and second_side == SIDE_A:
            return second_half, first_half

        first_signed = self.signed_distance_to_net(first_center)
        second_signed = self.signed_distance_to_net(second_center)
        if first_signed >= second_signed:
            return first_half, second_half
        return second_half, first_half

    def _build_net_zone_polygon(self, tolerance_px: float) -> np.ndarray:
        (x1, y1), (x2, y2) = self.net_line
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length <= 1e-9:
            return np.array([(x1, y1), (x2, y2), (x2, y2), (x1, y1)], dtype=np.float32)
        nx = -dy / length
        ny = dx / length
        offset = float(tolerance_px)
        points = np.array(
            [
                (x1 + nx * offset, y1 + ny * offset),
                (x2 + nx * offset, y2 + ny * offset),
                (x2 - nx * offset, y2 - ny * offset),
                (x1 - nx * offset, y1 - ny * offset),
            ],
            dtype=np.float32,
        )
        return points

    @staticmethod
    def _polygon_center(polygon: np.ndarray) -> Tuple[float, float]:
        center = np.mean(np.asarray(polygon, dtype=np.float32), axis=0)
        return float(center[0]), float(center[1])

    def contains_pixel(self, polygon: np.ndarray, point: Tuple[float, float]) -> bool:
        return bool(cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32), point, False) >= 0)
