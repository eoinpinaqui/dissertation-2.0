# Library imports
import numpy as np

# Local imports
from .object import GameObject

# Constants for the Missile class
MISSILE_SPEED = 12
HP = 1


# Class for missile in the game world
class Missile(GameObject):
    def __init__(self, x: int, y: int, angle: int):
        icon_path = './game/sprites/missile.png'
        super(Missile, self).__init__(icon_path, x, y, 0, 0, MISSILE_SPEED, MISSILE_SPEED, MISSILE_SPEED, HP)

        # Some custom stuff for the missile class
        points = np.zeros((4, 2))
        points[0] = (self.x - (self.icon_width // 4), self.y - (self.icon_height // 8))
        points[1] = (self.x - (self.icon_width // 4), self.y + (self.icon_height // 8))
        points[2] = (self.x + (self.icon_width // 4), self.y + (self.icon_height // 8))
        points[3] = (self.x + (self.icon_width // 4), self.y - (self.icon_height // 8))
        super().set_hit_box(points)
        super().rotate(angle)
