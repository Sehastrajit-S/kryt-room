import math

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import roomworld
from roomworld import FORWARD, INTERACT, TURN_LEFT, TURN_RIGHT, RoomEnv, Task
from roomworld.env import TURN_PER_STEP


def walk_action(env, field):
    """Follow a distance field: the reference solution showing the tasks can be solved."""
    w = env.unwrapped.world
    room = env.unwrapped.room
    best_turns, best_value = 0, math.inf
    for turns in range(-13, 14):          # only headings the body can land on exactly
        angle = w.heading + turns * TURN_PER_STEP
        sin, cos = math.sin(angle), math.cos(angle)
        if not (room.clear_at(w.x + sin * 0.06, w.z + cos * 0.06) and room.clear_at(w.x + sin * 0.15, w.z + cos * 0.15)):
            continue
        value = field.at(w.x + sin * 0.15, w.z + cos * 0.15) + 0.002 * abs(turns)
        if value < best_value:
            best_turns, best_value = turns, value
    if best_turns == 0:
        return FORWARD
    return TURN_LEFT if best_turns > 0 else TURN_RIGHT


@pytest.mark.parametrize("task", ["navigate", "caretaker"])
def test_passes_gymnasium_checks(task):
    check_env(RoomEnv(task=task), skip_render_check=True)


@pytest.mark.parametrize("env_id", ["RoomWorld/Navigate-v0", "RoomWorld/Caretaker-v0"])
def test_registered(env_id):
    env = gym.make(env_id)
    observation, _ = env.reset(seed=1)
    assert env.observation_space.contains(observation)


def test_every_place_and_switch_is_reachable():
    room = roomworld.Room()
    origin = room.walkable_points()[0]
    for point in [*room.places.values(), *room.lamps.values()]:
        assert room.clear_at(point["x"], point["z"])
        assert math.isfinite(room.distance_field(point["x"], point["z"]).at(*origin))


def test_random_policy_never_enters_furniture():
    env = RoomEnv()
    env.reset(seed=3)
    env.action_space.seed(3)
    for _ in range(4000):
        *_, terminated, truncated, _info = env.step(env.action_space.sample())
        assert env.room.clear_at(env.world.x, env.world.z)
        if terminated or truncated:
            env.reset()


def test_lidar_sees_a_known_wall():
    room = roomworld.Room()
    distance = room.raycast(0.0, 0.2, np.array([math.pi / 2]), 10.0)[0]   # +x from open floor
    assert distance == pytest.approx(3.0)


def test_navigate_is_solvable_from_any_seed():
    env = RoomEnv()
    for seed in range(30):
        env.reset(seed=seed)
        info, done = {}, False
        while not done:
            *_, terminated, truncated, info = env.step(walk_action(env, env.task.field))
            done = terminated or truncated
        assert info["success"], f"seed {seed}"


def test_caretaker_is_solvable():
    env = RoomEnv(task="caretaker")
    for seed in range(5):
        env.reset(seed=seed)
        info, done, total = {}, False, 0.0
        while not done:
            wrong = [n for n in env.world.lamps if env.task._wrong(env.world, n)]
            action = 0
            if wrong:
                def reach(name):
                    s = env.room.lamps[name]
                    return math.hypot(s["x"] - env.world.x, s["z"] - env.world.z)
                target = min(wrong, key=reach)
                switch = env.room.lamps[target]
                action = INTERACT if reach(target) <= 0.3 else walk_action(env, env.room.distance_field(switch["x"], switch["z"]))
            *_, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        assert info["success"], f"seed {seed}"


def test_custom_task():
    class StayNearWindow(Task):
        extra_obs = 1
        max_steps = 10

        def observe(self, world, room):
            return np.array([0.5], dtype=np.float32)

        def reward(self, world, room, events):
            return -abs(world.x + 2.25), False, {}

    env = RoomEnv(task=StayNearWindow())
    observation, _ = env.reset(seed=0)
    assert observation[-1] == pytest.approx(0.5)
    for step in range(10):
        *_, truncated, _info = env.step(0)
    assert truncated


def test_replay_file(tmp_path):
    import json
    env = RoomEnv(record=True)
    env.reset(seed=0)
    env.step(FORWARD)
    path = tmp_path / "runs" / "episode.json"
    env.save_replay(path)
    data = json.loads(path.read_text())
    assert data["dt"] == 0.1 and len(data["frames"]) == 2
    assert set(data["frames"][0]) == {"x", "z", "heading", "sun", "preset", "lamps"}
