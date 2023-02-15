# Library imports
import numpy as np

# Local imports
from .object import GameObject

# Constants for the Ship class
MAX_SPEED = 6
MIN_SPEED = 0
TURNING_ANGLE = 6
HP = 1
PLAYER_MISSILE_THRESHOLD = 25
ENEMY_MISSILE_THRESHOLD = 35


# Class for ships (agents) in the game world
class Ship(GameObject):
    def __init__(self, x: int, y: int, angle: int, speed: int, player: bool):
        icon_path = './game/sprites/player.png' if player else './game/sprites/enemy.png'
        super(Ship, self).__init__(icon_path, x, y, 0, TURNING_ANGLE, speed, MIN_SPEED, MAX_SPEED, HP)

        # Some custom stuff for the ship class
        self.missile_threshold = 0
        points = np.zeros((4, 2))
        points[0] = (self.x - (self.icon_width // 2), self.y - (self.icon_height // 4))
        points[1] = (self.x - (self.icon_width // 2), self.y + (self.icon_height // 4))
        points[2] = (self.x + (self.icon_width // 2), self.y + (self.icon_height // 4))
        points[3] = (self.x + (self.icon_width // 2), self.y - (self.icon_height // 4))
        super().set_hit_box(points)
        super().rotate(angle)

        self.missile_threshold = PLAYER_MISSILE_THRESHOLD if player else ENEMY_MISSILE_THRESHOLD
        self.missile_buffer = 0

    def move(self):
        super(Ship, self).move()
        self.missile_buffer += 1

    def can_fire_missile(self):
        if self.missile_buffer > self.missile_threshold:
            self.missile_buffer = 0
            return True
        return False
