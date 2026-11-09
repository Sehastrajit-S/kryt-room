"""Goals an agent can be trained on. Subclass Task to add your own."""
from __future__ import annotations

import numpy as np

DARK_BELOW = 0.35


def relative(world, x: float, z: float, scale: float = 6.0) -> tuple[float, float]:
    """A point in the agent's own frame: (ahead, to the left), scaled to about [-1, 1]."""
    dx, dz = x - world.x, z - world.z
    sin, cos = np.sin(world.heading), np.cos(world.heading)
    return (dx * sin + dz * cos) / scale, (dx * cos - dz * sin) / scale


class Task:
    """Defines what is observed beyond the base vector, what is rewarded, and when an episode ends."""

    extra_obs = 0
    max_steps = 400

    def reset(self, world, room, rng: np.random.Generator) -> None:
        pass

    def observe(self, world, room) -> np.ndarray:
        return np.zeros(self.extra_obs, dtype=np.float32)

    def reward(self, world, room, events: dict) -> tuple[float, bool, dict]:
        """Return (reward, terminated, info). events: {"collided": bool, "toggled": lamp name or None}."""
        raise NotImplementedError


class Navigate(Task):
    """Reach a randomly chosen named place. Rewards progress in walking distance."""

    extra_obs = 3
    max_steps = 400
    success_radius = 0.3

    def reset(self, world, room, rng):
        names = list(room.places)
        rng.shuffle(names)
        for name in names:
            place = room.places[name]
            self.field = room.distance_field(place["x"], place["z"])
            if self.field.at(world.x, world.z) > 1.0:
                break
        self.goal = name
        self.target = place
        self.last = self.field.at(world.x, world.z)

    def observe(self, world, room):
        ahead, left = relative(world, self.target["x"], self.target["z"])
        return np.array([ahead, left, min(self.last / 8.0, 1.0)], dtype=np.float32)

    def reward(self, world, room, events):
        distance = self.field.at(world.x, world.z)
        reward = 2.0 * (self.last - distance) - 0.005 - (0.05 if events["collided"] else 0.0)
        self.last = distance
        success = np.hypot(world.x - self.target["x"], world.z - self.target["z"]) < self.success_radius
        return reward + (5.0 if success else 0.0), bool(success), {"goal": self.goal, "success": bool(success)}


class Caretaker(Task):
    """Keep the lamps right as daylight changes: on when it is dark, off when it is bright."""

    extra_obs = 9
    max_steps = 600

    def reset(self, world, room, rng):
        self.sun_start, self.sun_end = rng.uniform(0.0, 1.0, 2)
        self.steps = 0
        world.sun = float(self.sun_start)
        for name in world.lamps:
            world.lamps[name] = bool(rng.integers(2))
        self.last = self._distance_to_work(world, room)

    def _wrong(self, world, name) -> bool:
        return world.lamps[name] != (world.sun < DARK_BELOW)

    def _distance_to_work(self, world, room) -> float:
        wrong = [room.lamps[n] for n in world.lamps if self._wrong(world, n)]
        if not wrong:
            return 0.0
        return min(room.distance_field(s["x"], s["z"]).at(world.x, world.z) for s in wrong)

    def observe(self, world, room):
        values = []
        for name in sorted(room.lamps):
            switch = room.lamps[name]
            values += [*relative(world, switch["x"], switch["z"]), float(self._wrong(world, name))]
        return np.array(values, dtype=np.float32)

    def reward(self, world, room, events):
        self.steps += 1
        world.sun = float(self.sun_start + (self.sun_end - self.sun_start) * min(self.steps / self.max_steps, 1.0))
        wrong = sum(self._wrong(world, n) for n in world.lamps)
        reward = -0.01 * wrong - (0.05 if events["collided"] else 0.0)
        if events["toggled"]:
            reward += 1.0 if not self._wrong(world, events["toggled"]) else -1.0
        distance = self._distance_to_work(world, room)
        if distance and self.last:
            reward += self.last - distance
        self.last = distance
        return reward, False, {"lamps_wrong": int(wrong), "success": wrong == 0}


TASKS = {"navigate": Navigate, "caretaker": Caretaker}
