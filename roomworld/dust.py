"""Dust that behaves like a lived-in room: it starts where dirt really comes from and keeps changing.

Each 25 cm floor square holds a level from 0 (clean) to 1 (filthy). Settings live under "dust" in room.json.
"""
from __future__ import annotations

import math

import numpy as np

CELL = 0.25
DIRTY_ABOVE = 0.1


class Dust:
    def __init__(self, room):
        self.room = room
        self.config = room.dust_config
        xs = np.arange(-room.half_x + CELL / 2, room.half_x, CELL)
        zs = np.arange(-room.half_z + CELL / 2, room.half_z, CELL)
        self.nx, self.nz = len(xs), len(zs)
        grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")
        self.mask = np.array([[room.clear_at(x, z) for z in zs] for x in xs])
        self.x, self.z = grid_x[self.mask], grid_z[self.mask]
        self.levels = np.zeros(len(self.x))
        self.weights = np.zeros(len(self.x))
        self.edges = self._edge_field()
        self.routes = [self._route_field(route) for route in self.config["traffic"]["routes"]]
        self.retain = self._retain_field()
        self.active_routes: list[np.ndarray] = []

    def _edge_field(self) -> np.ndarray:
        """Dust settles along walls and furniture."""
        edges = self.config["edges"]
        angles = np.arange(16) * (2 * math.pi / 16)
        gaps = np.array([self.room.raycast(x, z, angles, 3.0).min() - self.room.radius for x, z in zip(self.x, self.z)])
        return edges["strength"] * np.exp(-np.maximum(gaps, 0.0) / edges["width"])

    def _segment_distance(self, ax: float, az: float, bx: float, bz: float) -> np.ndarray:
        dx, dz = bx - ax, bz - az
        t = np.clip(((self.x - ax) * dx + (self.z - az) * dz) / (dx * dx + dz * dz), 0.0, 1.0)
        return np.hypot(self.x - (ax + t * dx), self.z - (az + t * dz))

    def _route_field(self, path: list) -> np.ndarray:
        """Foot traffic: squares close to a walking route, given as a list of points."""
        traffic = self.config["traffic"]
        distance = np.min([self._segment_distance(*a, *b) for a, b in zip(path, path[1:])], axis=0)
        return traffic["strength"] * np.exp(-((distance / traffic["width"]) ** 2))

    def _retain_field(self) -> np.ndarray:
        """Surfaces such as the rug hold on to more dirt."""
        factor = np.ones(len(self.x))
        for zone in self.config["retain"]:
            x0, x1, z0, z1 = zone["rect"]
            factor[(self.x >= x0) & (self.x <= x1) & (self.z >= z0) & (self.z <= z1)] = zone["factor"]
        return factor

    def _source_field(self, source: dict, rng: np.random.Generator) -> np.ndarray:
        if "from" in source:
            distance = self._segment_distance(*source["from"], *source["to"])
            spread = source["width"]
        else:
            jitter = source.get("jitter", 0.0)
            cx = source["x"] + rng.uniform(-jitter, jitter)
            cz = source["z"] + rng.uniform(-jitter, jitter)
            distance = np.hypot(self.x - cx, self.z - cz)
            spread = source["radius"]
        return source["strength"] * np.exp(-((distance / spread) ** 2))

    def reset(self, rng: np.random.Generator) -> None:
        """Lay out a new day's dirt: how busy it was, and which sources were active."""
        low, high = self.config["busy_day_range"]
        busy = rng.uniform(low, high)
        chance = self.config["traffic"]["chance"]
        self.active_routes = [route for route in self.routes if rng.random() < chance]
        weights = self.config["background"] + self.edges + sum(self.active_routes, np.zeros(len(self.x)))
        for source in self.config["sources"]:
            if rng.random() < source.get("chance", 1.0):
                weights = weights + self._source_field(source, rng)
        self.weights = weights * self.retain
        noise = self.config["noise"] * rng.normal(size=len(self.x))
        self.levels = np.clip(busy * self.weights + noise, 0.0, 1.0)

    def step(self, seconds: float, rng: np.random.Generator) -> None:
        """Dust keeps settling, and now and then someone walks through or something spills."""
        minutes = seconds / 60.0
        self.levels = np.minimum(self.levels + self.config["settle_per_minute"] * minutes * self.weights, 1.0)
        if rng.random() < self.config["events_per_minute"] * minutes:
            if self.active_routes and rng.random() < 0.5:
                addition = 0.5 * self.active_routes[rng.integers(len(self.active_routes))]
            else:
                index = rng.integers(len(self.x))
                distance = np.hypot(self.x - self.x[index], self.z - self.z[index])
                addition = 0.8 * np.exp(-((distance / 0.35) ** 2))
            self.levels = np.minimum(self.levels + addition, 1.0)

    def clean(self, x: float, z: float, radius: float, power: float = 0.8) -> float:
        """Remove up to `power` from squares within `radius`, and return how much dirt came up."""
        reached = (self.x - x) ** 2 + (self.z - z) ** 2 < radius**2
        removed = np.minimum(self.levels[reached], power)
        self.levels[reached] -= removed
        return float(removed.sum())

    def sense(self, x: float, z: float, heading: float, radius: float = 1.5, sectors: int = 8) -> np.ndarray:
        """Dirt within `radius`, split into directions around the given heading, each from 0 to 1."""
        dx, dz = self.x - x, self.z - z
        near = dx**2 + dz**2 < radius**2
        readings = np.zeros(sectors, dtype=np.float32)
        if near.any():
            angles = (np.arctan2(dx[near], dz[near]) - heading) % (2 * math.pi)
            bins = np.floor(angles / (2 * math.pi) * sectors).astype(int) % sectors
            readings = np.bincount(bins, weights=self.levels[near], minlength=sectors).astype(np.float32)
        return np.minimum(readings / 4.0, 1.0)

    def dirty_fraction(self) -> float:
        return float((self.levels > DIRTY_ABOVE).mean())

    def total(self) -> float:
        return float(self.levels.sum())

    def layout(self) -> dict:
        return {"nx": self.nx, "nz": self.nz, "cell": CELL}

    def encode(self) -> str:
        """The whole floor as one digit per square (0-9), row by row, for the 3D view."""
        grid = np.zeros((self.nx, self.nz))
        grid[self.mask] = self.levels
        return ((grid.T.ravel() * 9.999).astype(np.uint8) + 48).tobytes().decode("ascii")
