# Library imports
import gym
import numpy as np
import shapely
import cv2
import math

# Local imports
from .ship import Ship
from .missile import Missile
from .object import GameObject

# Constants for the SinglePlayerGame class
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
OBSERVATION_SHAPE = (80, 80)
SPAWN_ENEMIES_INTERVAL = 100
ENEMY_MARGIN = 50
ENEMY_SPEED = 2
NOOP = 0
ACCELERATE = 1
DECELERATE = 2
TURN_LEFT = 3
TURN_RIGHT = 4
FIRE_MISSILE = 5
REWARD = 1
PENALTY = -10


# Class for a single player version of the environment
class SinglePlayerGame(gym.Env):
    def __init__(self):
        # Define the observation space
        self.observation_space = gym.spaces.Box(low=np.zeros(OBSERVATION_SHAPE), high=np.full(OBSERVATION_SHAPE, 255), dtype=np.uint8)
        self.arena = shapely.geometry.polygon.Polygon([(0, 0), (WINDOW_WIDTH, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, WINDOW_HEIGHT)])

        # Define the action space
        self.action_space = gym.spaces.Discrete(6, )
        print(self.action_space)

        # Create a canvas to draw the game on
        self.canvas = np.full((WINDOW_WIDTH, WINDOW_HEIGHT, 3), 255, dtype=np.uint8)

        # Create the player, enemies and missiles
        self.player = Ship(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, 90, 0, player=True)
        self.enemies = []
        self.missiles = []

        # Set time to 0
        self.time = 0

        # Draw the game on the canvas
        self.draw_game_on_canvas()

    def draw_element_on_canvas(self, element: GameObject):
        if self.arena.covers(element.hit_box):
            (minx, miny, maxx, maxy) = element.hit_box.bounds
            (minx, miny, maxx, maxy) = (int(minx), int(miny), int(maxx), int(maxy))
            self.canvas[WINDOW_HEIGHT - maxy:WINDOW_HEIGHT - miny, minx:maxx] = \
                element.icon[
                element.padded_icon_height - (element.padded_icon_height - (maxy - miny)) // 2 - (maxy - miny):
                element.padded_icon_height - (element.padded_icon_height - (maxy - miny)) // 2,
                (element.padded_icon_width - (maxx - minx)) // 2:
                (element.padded_icon_width - (maxx - minx)) // 2 + (maxx - minx)
                ]

    def draw_game_on_canvas(self):
        self.canvas = np.full((WINDOW_WIDTH, WINDOW_HEIGHT, 3), 255, dtype=np.uint8)
        self.draw_element_on_canvas(self.player)
        for enemy in self.enemies:
            self.draw_element_on_canvas(enemy)
        for missile in self.missiles:
            self.draw_element_on_canvas(missile)

    def reset(self):
        # Create a canvas to draw the game on
        self.canvas = np.full((WINDOW_WIDTH, WINDOW_HEIGHT, 3), 255, dtype=np.uint8)

        # Create the player, enemies and missiles
        self.player = Ship(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, 90, 0, player=True)
        self.enemies = []
        self.missiles = []

        # Set time back to 0
        self.time = 0

        # Draw the game on the canvas
        self.draw_game_on_canvas()

        return self.preprocess_frame()

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array'], 'Invalid mode, must be either "human" or "rgb_array"'
        if mode == "human":
            cv2.imshow("Game", self.preprocess_frame())
            cv2.waitKey(25)
        elif mode == "rgb_array":
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()

    def preprocess_frame(self):
        resized = cv2.resize(self.canvas, OBSERVATION_SHAPE)
        grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return grayscale

    def step(self, action):
        done = False
        self.time += 1

        # Apply the chosen action
        assert self.action_space.contains(action), "Invalid Action"
        if action == TURN_LEFT:
            self.player.rotate_left()
        elif action == TURN_RIGHT:
            self.player.rotate_right()
        elif action == ACCELERATE:
            self.player.accelerate()
        elif action == DECELERATE:
            self.player.decelerate()
        elif action == FIRE_MISSILE and self.player.can_fire_missile():
            x_off = round(np.cos(np.radians(self.player.angle)) * (self.player.speed + 16))
            y_off = round(np.sin(np.radians(self.player.angle)) * (self.player.speed + 16))
            x_off *= 2
            y_off *= 2
            self.missiles.append(Missile(self.player.x + x_off, self.player.y + y_off, self.player.angle))

        # Update the state of all elements in the game world and draw them on the canvas
        self.player.move()
        if self.time % SPAWN_ENEMIES_INTERVAL == 0 and self.time != 0:
            self.enemies.append(Ship(ENEMY_MARGIN, WINDOW_HEIGHT // 2, 0, ENEMY_SPEED, player=False))
            self.enemies.append(Ship(WINDOW_WIDTH - ENEMY_MARGIN, WINDOW_HEIGHT // 2, 180, ENEMY_SPEED, player=False))

        # Update the enemies
        for enemy in self.enemies:
            # Angle the enemy towards the player and move them
            enemy_angle = enemy.angle % 360
            angle_to_player = math.degrees(math.atan2(self.player.y - enemy.y, self.player.x - enemy.x))
            if angle_to_player < 0:
                angle_to_player = 360 + angle_to_player
            if int(enemy_angle) != int(angle_to_player):
                turn_left_start = angle_to_player - 180
                if turn_left_start >= 0:
                    if turn_left_start < enemy_angle < angle_to_player:
                        enemy.rotate(min(enemy.turning_angle, angle_to_player - enemy_angle))
                    else:
                        enemy.rotate(max(-enemy.turning_angle, angle_to_player - enemy_angle))
                else:
                    turn_left_start = 360 + turn_left_start
                    if turn_left_start < enemy_angle or enemy_angle < angle_to_player:
                        enemy.rotate(min(enemy.turning_angle, angle_to_player - enemy_angle))
                    else:
                        enemy.rotate(max(-enemy.turning_angle, angle_to_player - enemy_angle))
            enemy.move()

            # Fire a missile if possible
            if enemy.can_fire_missile():
                x_off = round(np.cos(np.radians(enemy.angle)) * (enemy.speed + 16))
                y_off = round(np.sin(np.radians(enemy.angle)) * (enemy.speed + 16))
                x_off *= 2
                y_off *= 2
                self.missiles.append(Missile(enemy.x + x_off, enemy.y + y_off, enemy.angle))

            # Check for collisions
            for other_enemy in self.enemies:
                if other_enemy != enemy and enemy.hit_box.intersects(other_enemy.hit_box):
                    self.enemies.remove(enemy)
                    self.enemies.remove(other_enemy)

            if not self.arena.covers(enemy.hit_box):
                self.enemies.remove(enemy)

            if self.player.hit_box.intersects(enemy.hit_box):
                self.player.decrease_hp()

        # Update the missiles
        for missile in self.missiles:
            missile.move()

            # Check for collisions
            for other_missile in self.missiles:
                if missile != other_missile and other_missile.hit_box.intersects(missile.hit_box):
                    self.missiles.remove(missile)
                    self.missiles.remove(other_missile)

            for enemy in self.enemies:
                if enemy.hit_box.intersects(missile.hit_box):
                    self.enemies.remove(enemy)
                    self.missiles.remove(missile)

            if self.player.hit_box.intersects(missile.hit_box):
                self.player.decrease_hp()

        self.draw_game_on_canvas()

        # Calculate the reward
        reward = REWARD

        if not self.arena.covers(self.player.hit_box) or self.player.hp <= 0:
            done = True
            reward = PENALTY

        return self.preprocess_frame(), reward, done, {}
