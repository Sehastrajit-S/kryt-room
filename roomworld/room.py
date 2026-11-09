"""Room geometry shared with the 3D viewer: loaded from world/room.json."""
from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np

ROOM_JSON = Path(__file__).resolve().parents[1] / "world" / "room.json"
CELL = 0.1


class DistanceField:
    """Walking distance to one point, around furniture."""

    def __init__(self, room: "Room", values: np.ndarray):
        self.room = room
        self.values = values

    def at(self, x: float, z: float) -> float:
        i = (x + self.room.half_x) / CELL
        j = (z + self.room.half_z) / CELL
        best = math.inf
        for ci in (math.floor(i), math.floor(i) + 1):
            for cj in (math.floor(j), math.floor(j) + 1):
                if 0 <= ci < self.values.shape[0] and 0 <= cj < self.values.shape[1]:
                    corner = self.values[ci, cj]
                    if math.isfinite(corner):
                        best = min(best, corner + math.hypot(i - ci, j - cj) * CELL)
        return best


class Room:
    def __init__(self, path: Path | str = ROOM_JSON, radius: float = 0.24):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.half_x = float(data["floor"]["halfX"])
        self.half_z = float(data["floor"]["halfZ"])
        self.obstacles = np.array(list(data["obstacles"].values()), dtype=np.float64)
        self.places = data["places"]
        self.lamps = data["lamps"]
        self.switch_reach = float(data["switchReach"])
        self.dust_config = data["dust"]
        self.radius = radius
        self._nx = round(self.half_x * 2 / CELL) + 1
        self._nz = round(self.half_z * 2 / CELL) + 1
        self.walkable = np.array(
            [[self.clear_at(*self._cell_point(i, j)) for j in range(self._nz)] for i in range(self._nx)]
        )
        self._fields: dict[tuple[float, float], DistanceField] = {}

    def _cell_point(self, i: int, j: int) -> tuple[float, float]:
        return -self.half_x + i * CELL, -self.half_z + j * CELL

    def clear_at(self, x: float, z: float) -> bool:
        r = self.radius
        if abs(x) > self.half_x - r or abs(z) > self.half_z - r:
            return False
        o = self.obstacles
        hit = (x > o[:, 0] - r) & (x < o[:, 1] + r) & (z > o[:, 2] - r) & (z < o[:, 3] + r)
        return not bool(hit.any())

    def walkable_points(self) -> np.ndarray:
        i, j = np.nonzero(self.walkable)
        return np.stack([-self.half_x + i * CELL, -self.half_z + j * CELL], axis=1)

    def distance_field(self, x: float, z: float) -> DistanceField:
        key = (round(x, 3), round(z, 3))
        if key not in self._fields:
            self._fields[key] = DistanceField(self, self._dijkstra(x, z))
        return self._fields[key]

    def _dijkstra(self, x: float, z: float) -> np.ndarray:
        points = self.walkable_points()
        nearest = points[np.argmin(np.hypot(points[:, 0] - x, points[:, 1] - z))]
        start = (round((nearest[0] + self.half_x) / CELL), round((nearest[1] + self.half_z) / CELL))
        values = np.full((self._nx, self._nz), np.inf)
        values[start] = 0.0
        queue = [(0.0, start)]
        while queue:
            cost, (i, j) = heapq.heappop(queue)
            if cost > values[i, j]:
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if (di, dj) == (0, 0) or not (0 <= ni < self._nx and 0 <= nj < self._nz):
                        continue
                    if not (self.walkable[ni, nj] and self.walkable[ni, j] and self.walkable[i, nj]):
                        continue
                    total = cost + math.hypot(di, dj) * CELL
                    if total < values[ni, nj]:
                        values[ni, nj] = total
                        heapq.heappush(queue, (total, (ni, nj)))
        return values

    def raycast(self, x: float, z: float, angles: np.ndarray, max_distance: float) -> np.ndarray:
        """Distance along each heading to the nearest wall or piece of furniture."""
        dx, dz = np.sin(angles), np.cos(angles)
        with np.errstate(divide="ignore", invalid="ignore"):
            wall_x = np.where(dx > 0, (self.half_x - x) / dx, np.where(dx < 0, (-self.half_x - x) / dx, np.inf))
            wall_z = np.where(dz > 0, (self.half_z - z) / dz, np.where(dz < 0, (-self.half_z - z) / dz, np.inf))
            distances = np.minimum(wall_x, wall_z)

            o = self.obstacles
            tx0 = (o[:, 0, None] - x) / dx[None, :]
            tx1 = (o[:, 1, None] - x) / dx[None, :]
            tz0 = (o[:, 2, None] - z) / dz[None, :]
            tz1 = (o[:, 3, None] - z) / dz[None, :]
            near = np.maximum(np.minimum(tx0, tx1), np.minimum(tz0, tz1))
            far = np.minimum(np.maximum(tx0, tx1), np.maximum(tz0, tz1))
        hits = np.where((far >= near) & (far > 0) & np.isfinite(near), np.maximum(near, 0.0), np.inf)
        return np.minimum(np.minimum(distances, hits.min(axis=0)), max_distance)
