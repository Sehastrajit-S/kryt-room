import gymnasium as gym

from .env import FORWARD, IDLE, INTERACT, TURN_LEFT, TURN_RIGHT, RoomEnv
from .room import Room
from .tasks import Caretaker, Navigate, Task

gym.register(id="RoomWorld/Navigate-v0", entry_point="roomworld.env:RoomEnv", kwargs={"task": "navigate"})
gym.register(id="RoomWorld/Caretaker-v0", entry_point="roomworld.env:RoomEnv", kwargs={"task": "caretaker"})

__all__ = ["RoomEnv", "Room", "Task", "Navigate", "Caretaker", "IDLE", "FORWARD", "TURN_LEFT", "TURN_RIGHT", "INTERACT"]
