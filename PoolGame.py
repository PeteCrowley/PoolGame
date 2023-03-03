import random
from typing import Optional, Union, Tuple
import numpy as np
import arcade
import math
import arcade.gui
import pickle

from gym.core import ObsType, ActType
import gym
from gym import spaces
from gym.envs.registration import register


register(
    id='PoolGame-v0',
    entry_point='PoolGame:GymMiniGame',
    max_episode_steps=300,
)


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Pool"

TABLE_WIDTH = SCREEN_WIDTH * 7/8
TABLE_HEIGHT = TABLE_WIDTH / 2

SCALE = SCREEN_WIDTH / 8            # 1 How many pixels equals 1 foot

POOL_BALL_WIDTH = SCALE * 0.1875
POOL_BALL_SCALE = POOL_BALL_WIDTH / 150
POOL_BALL_MASS = 0.16

TABLE_FRICTION = 0.1
CUSHION_BOUNCINESS = 0.8
GRAVITY_CONSTANT = 9.81
ACCELERATION_FROM_FRICTION = TABLE_FRICTION * GRAVITY_CONSTANT

FPS = 40

X_Padding = TABLE_WIDTH * 0.064
Y_Padding = TABLE_HEIGHT * 0.111
dist_to_end_rail = TABLE_HEIGHT * 0.0657

LEFT_RAIL = SCREEN_WIDTH / 2 - TABLE_WIDTH / 2 + X_Padding
RIGHT_RAIL = SCREEN_WIDTH / 2 + TABLE_WIDTH / 2 - X_Padding
TOP_RAIL = SCREEN_HEIGHT / 2 + TABLE_HEIGHT / 2 - Y_Padding
BOTTOM_RAIL = SCREEN_HEIGHT / 2 - TABLE_HEIGHT / 2 + Y_Padding

MAX_VELOCITY = 15
NUM_BALLS = 9


def true_inverse_tangent(x, y):
    if x == 0:
        theta = math.pi / 2 if y > 0 else 3 * math.pi / 2
    elif y == 0:
        theta = 0 if x > 0 else math.pi
    else:
        theta = math.atan(y / x)
        if x < 0:
            theta += math.pi
    return theta


def get_pocket_locations():
    pixels = []
    pocket_radius = 0.020 * TABLE_WIDTH
    pocket_centers = [(LEFT_RAIL - 0.006857 * TABLE_WIDTH, BOTTOM_RAIL - TABLE_HEIGHT * .005286),
                      (LEFT_RAIL - 0.006857 * TABLE_WIDTH, TOP_RAIL + TABLE_HEIGHT * .005286),
                      ((LEFT_RAIL + RIGHT_RAIL) / 2, TOP_RAIL + TABLE_HEIGHT * .03),
                      (RIGHT_RAIL + 0.005857 * TABLE_WIDTH, TOP_RAIL + TABLE_HEIGHT * .005286),
                      (RIGHT_RAIL + 0.005857 * TABLE_WIDTH, BOTTOM_RAIL - TABLE_HEIGHT * .005286),
                      ((LEFT_RAIL + RIGHT_RAIL) / 2, BOTTOM_RAIL - TABLE_HEIGHT * .03)]
    ranges = [(-20, 110), (-110, 20), (160, 380), (160, 290), (70, 200), (-20, 200)]
    for i in range(6):
        pixels.append([])
        for theta in range(ranges[i][0], ranges[i][1], 5):
            point = (round(pocket_centers[i][0] + pocket_radius * math.cos(math.radians(theta))),
                     round(pocket_centers[i][1] + pocket_radius * math.sin(math.radians(theta))))
            pixels[i].append(point)
    return pixels


class PoolBallSprite(arcade.Sprite):
    def __init__(self, number, mass, start_x=SCREEN_WIDTH / 2, start_y=SCREEN_HEIGHT / 2):
        file_name = "Sprites/Balls/" + str(number) + "ball.png"
        super().__init__(file_name, scale=POOL_BALL_SCALE, center_x=start_x, center_y=start_y,
                         hit_box_algorithm="Detailed")
        self.mass: float = mass
        self.number: int = number
        self.is_colliding_with = []
        self._position: np.array = np.array([start_x+0., start_y+0.])
        self.velocity: np.array = np.array([0., 0.])

    def is_cue_ball(self):
        """
        Checks if the ball is the cue ball for the table
        :return: True if the ball is the cue ball, otherwise false
        """
        return self.number == 0

    def is_solid(self):
        """
        Checks if the ball is a solid
        :return: True if the ball is a solid, otherwise false
        """
        return 0 < self.number < 8

    def is_stripe(self):
        """
        Checks if the ball is a stripe
        :return: True if the ball is a stipe, otherwise false
        """
        return 8 < self.number < 16

    def update(self):
        """
        Update the sprite.
        """
        self.position = self.position + self.velocity * (SCALE / FPS)
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude < 0.05:
            self.stop()
        else:
            percent_change = (velocity_magnitude - ACCELERATION_FROM_FRICTION / FPS) / velocity_magnitude
            percent_change = max(0, percent_change)
            self.velocity = percent_change * self.velocity
        self.handle_rail_collision()
        for ball in self.is_colliding_with:
            if not arcade.check_for_collision(self, ball):
                self.is_colliding_with.remove(ball)

    def handle_rail_collision(self):
        # Colliding With Rails
        rail_coll = False
        vert_change = False
        horiz_change = False
        if self.center_x - self.width / 2 < LEFT_RAIL and self.change_x < 0:
            rail_coll = True
            vert_change = True
        if self.center_x + self.width / 2 > RIGHT_RAIL and self.change_x > 0:
            vert_change = True
            rail_coll = True
        if self.center_y + self.width / 2 > TOP_RAIL and self.change_y > 0:
            horiz_change = True
            rail_coll = True
        if self.center_y - self.width / 2 < BOTTOM_RAIL and self.change_y < 0:
            horiz_change = True
            rail_coll = True
        if vert_change:
            self.change_x = -self.change_x
        if horiz_change:
            self.change_y = -self.change_y
        if rail_coll:
            self.change_x *= CUSHION_BOUNCINESS
            self.change_y *= CUSHION_BOUNCINESS

    # Sometimes a little buggy because of FPS update lag
    def handle_ball_collision(self, ball):
        if ball in self.is_colliding_with:
            return
        self.is_colliding_with.append(ball)
        r1, r2 = self.position, ball.position
        d = np.linalg.norm(r1 - r2) ** 2
        v1, v2 = self.velocity, ball.velocity
        u1 = v1 - np.dot(v1 - v2, r1 - r2) / d * (r1 - r2)
        u2 = v2 - np.dot(v2 - v1, r2 - r1) / d * (r2 - r1)
        self.velocity = u1
        ball.velocity = u2

    @property
    def position(self) -> np.array:
        """
        Get the center x and y coordinates of the sprite.

        Returns:
            (center_x, center_y)
        """
        return self._position

    @position.setter
    def position(self, new_value: np.array):
        """
        Set the center x and y coordinates of the sprite.

        :param Point new_value: New position.
        """
        if new_value[0] != self._position[0] or new_value[1] != self._position[1]:
            self.clear_spatial_hashes()
            self._point_list_cache = None
            self._position = new_value
            self.add_spatial_hashes()

            for sprite_list in self.sprite_lists:
                sprite_list.update_location(self)


class PoolGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, title, num_balls):
        super().__init__(width, height, title, update_rate=1 / FPS)
        arcade.set_background_color(arcade.color.AMAZON)
        self.pool_ball_list = None
        self.pool_table_image = None
        self.power_meter_image = None
        self.pocket_points: [(int, int)] = None
        self.power = None
        self.aim_slope = None
        self.aim_changing = False
        self.power_changing = False
        self.mouse_position = None

        self.uimanager = arcade.gui.UIManager()
        self.uimanager.enable()
        self.shot_in_prog = None
        self.num_balls = num_balls

    def variable_setup(self):
        self.power = MAX_VELOCITY / 2
        self.aim_slope = np.array([1, 0])
        self.aim_changing = False
        self.power_changing = False
        self.shot_in_prog = False
        self.pool_ball_list = arcade.SpriteList(use_spatial_hash=True)
        for i in range(self.num_balls + 1):
            start_x = random.randrange(int(LEFT_RAIL + POOL_BALL_WIDTH), int(RIGHT_RAIL - POOL_BALL_WIDTH))
            start_y = random.randrange(int(BOTTOM_RAIL + POOL_BALL_WIDTH), int(TOP_RAIL - POOL_BALL_WIDTH))
            ball = PoolBallSprite(i, POOL_BALL_MASS, start_x, start_y)
            while arcade.check_for_collision_with_list(ball, self.pool_ball_list):
                ball.center_x = random.randrange(int(LEFT_RAIL + POOL_BALL_WIDTH), int(RIGHT_RAIL - POOL_BALL_WIDTH))
                ball.center_y = random.randrange(int(BOTTOM_RAIL + POOL_BALL_WIDTH), int(TOP_RAIL - POOL_BALL_WIDTH))
            self.pool_ball_list.append(ball)
        self.pocket_points = get_pocket_locations()

    def setup(self, event=None):
        self.variable_setup()
        """ Set up the game variables. Call to re-start the game. """
        self.pool_table_image = arcade.load_texture("Sprites/PoolTable2.png")
        self.power_meter_image = arcade.load_texture("Sprites/PowerMeter.png")


        shoot_button = arcade.gui.UIFlatButton(text="Shoot", x=3 * SCREEN_WIDTH / 8, y=SCREEN_HEIGHT / 15,
                                               width=SCREEN_WIDTH / 4)
        shoot_button.on_click = self.shoot
        reset_button = arcade.gui.UIFlatButton(text="Reset", x=13 * SCREEN_WIDTH / 16, y=7 * SCREEN_HEIGHT / 8,
                                               width=SCREEN_WIDTH / 8)
        reset_button.on_click = self.setup
        self.uimanager.add(shoot_button)
        self.uimanager.add(reset_button)

    def on_draw(self):
        """
        Render the screen.
        """

        # This command should happen before we start drawing. It will clear
        # the screen to the background color, and erase what we drew last frame.
        self.clear()

        arcade.draw_lrwh_rectangle_textured(SCREEN_WIDTH / 2 - TABLE_WIDTH / 2, SCREEN_HEIGHT / 2 - TABLE_HEIGHT / 2,
                                            TABLE_WIDTH, TABLE_HEIGHT, self.pool_table_image)
        arcade.draw_lrwh_rectangle_textured(SCREEN_WIDTH / 4, SCREEN_HEIGHT - SCREEN_HEIGHT / 8, SCREEN_WIDTH / 2,
                                            SCREEN_HEIGHT / 12, self.power_meter_image)

        # Call draw() on all your sprite lists below
        self.uimanager.draw()
        self.pool_ball_list.draw()

        # Deal with aim and power
        if not self.shot_in_prog and self.aim_changing:
            self.change_aim()

        if not self.shot_in_prog and self.power_changing:
            self.change_power()

        if not self.shot_in_prog:
            self.make_aim_line()

        self.make_power_line()

    def on_update(self, delta_time):
        """
        Game Logic
        """
        self.pool_ball_list.update()
        self.shot_in_prog = self.shot_in_progress()

        if not self.shot_in_prog:
            return
        for i in range(len(self.pool_ball_list)).__reversed__():
            ball = self.pool_ball_list[i]
            for x in range(0, i):
                if arcade.check_for_collision(ball, self.pool_ball_list[x]):
                    ball.handle_ball_collision(self.pool_ball_list[x])
            for pocket in self.pocket_points:
                for point in pocket:
                    if ball.collides_with_point(point):
                        self.handle_made_ball(ball)
                        break

    def handle_made_ball(self, ball):
        if ball.is_cue_ball():
            self.setup()
        self.pool_ball_list.remove(ball)

    def shot_in_progress(self):
        for ball in self.pool_ball_list:
            if ball.change_x != 0:
                return True
            if ball.change_y != 0:
                return True
        return False

    def on_key_release(self, key, key_modifiers):
        """
        Called whenever the user lets off a previously pressed key.
        """
        if key == arcade.key.R:
            self.setup()
        elif key == arcade.key.A:
            self.aim_changing = not self.aim_changing
        elif key == arcade.key.P:
            self.power_changing = not self.power_changing
        elif key == arcade.key.S:
            self.shoot()
        elif key == arcade.key.E:
            pos = self.export_ball_positions()
            print(pos)
            print(pos.shape)

    def shoot(self, event=None):
        self.shot_in_prog = True
        self.aim_changing = False
        self.power_changing = False
        cue_ball = self.pool_ball_list[0]
        cue_ball.velocity = cue_ball.velocity + self.power * self.aim_slope

    def on_mouse_press(self, x, y, button, key_modifiers):
        """
        Called when the user presses a mouse button.
        """
        if self.shot_in_prog:
            return
        if SCREEN_WIDTH / 4 < x < 3 * SCREEN_WIDTH / 4 and 7 * SCREEN_HEIGHT / 8 < y < SCREEN_HEIGHT * 23 / 24:
            self.power_changing = not self.power_changing
        elif LEFT_RAIL < x < RIGHT_RAIL and BOTTOM_RAIL < y < TOP_RAIL:
            self.aim_changing = not self.aim_changing

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int):
        self.mouse_position = np.array([x, y])

    def change_aim(self):
        start = self.pool_ball_list[0].position
        end = self.mouse_position
        if end is not None:
            slope = end - start
            slope = slope / np.linalg.norm(slope)
            self.aim_slope = slope

    def make_aim_line(self):
        start = self.pool_ball_list[0].position
        x, y = start
        while LEFT_RAIL < x < RIGHT_RAIL and BOTTOM_RAIL < y < TOP_RAIL:
            x += self.aim_slope[0]
            y += self.aim_slope[1]
        arcade.draw_line(start[0], start[1], x, y, color=arcade.color.WHITE)

    def change_power(self):
        scale_factor = (SCREEN_WIDTH / 2) / MAX_VELOCITY
        x, y = self.mouse_position
        if SCREEN_WIDTH / 4 < x < 3 * SCREEN_WIDTH / 4 and 7 * SCREEN_HEIGHT / 8 < y < SCREEN_HEIGHT * 23 / 24:
            self.power = (x - SCREEN_WIDTH / 4) / scale_factor

    def make_power_line(self):
        scale_factor = (SCREEN_WIDTH / 2) / MAX_VELOCITY
        x = scale_factor * self.power + SCREEN_WIDTH / 4
        arcade.draw_line(x, SCREEN_HEIGHT * 7 / 8, x, SCREEN_HEIGHT * 23 / 24, color=arcade.color.WHITE)

    def export_ball_positions(self):
        positions = []
        extras = 0
        for i in range(len(self.pool_ball_list)):
            ball = self.pool_ball_list[i]
            if ball.number != i + extras:
                positions.append(np.array([0, 0]))
                extras += 1
            positions.append(ball.position)
        while len(positions) != self.num_balls + 1:
            positions.append(np.array([0, 0]))
        return np.array(positions).astype(float)


class MyTestGame(PoolGame):
    def __init__(self, width, height, title, num_balls):
        super().__init__(width, height, title, num_balls)

    def set_positions(self):
        self.pool_ball_list[0].position = np.array([250, 200])
        self.pool_ball_list[1].position = np.array([232, 300])
        self.aim_slope = np.array([0, 1])

    def on_key_release(self, key, key_modifiers):
        super().on_key_release(key, key_modifiers)
        if key == arcade.key.K:
            self.set_positions()


class MiniGame(PoolGame):
    def __init__(self, width, height, title, num_balls, shots=5):
        super().__init__(width, height, title, num_balls)
        self.max_shots = shots
        self.shots_left = None
        self.shots_left_text = None
        self.game_over = None
        self.game_over_text = None
        self.score = None
        self.score_text = None

    def variable_setup(self):
        super().variable_setup()
        self.shots_left = self.max_shots
        self.game_over = False
        self.score = 0
        self.pool_ball_list[0].position = np.array([450, 300])
        self.pool_ball_list[1].position = np.array([600, 225])

    def setup(self, event=None):
        super().setup(event=event)
        self.variable_setup()
        self.shots_left_text = arcade.Text(f"Shots Left: {self.shots_left}", start_x=SCREEN_WIDTH / 16,
                                           start_y=1 * SCREEN_HEIGHT / 10,
                                           width=SCREEN_WIDTH // 8, font_size=20)
        self.game_over_text = arcade.Text(f"Game Over!\nFinal Score: {self.score}", start_x=3 * SCREEN_WIDTH / 8,
                                          start_y=5 * SCREEN_HEIGHT / 8, width=SCREEN_WIDTH // 4,
                                          font_size=20, multiline=True)
        self.score_text = arcade.Text(f"Score: {self.score}", start_x=SCREEN_WIDTH / 16, start_y=9 * SCREEN_HEIGHT / 10,
                                      width=SCREEN_WIDTH // 8, font_size=20)

    def on_update(self, delta_time):
        if self.game_over:
            return
        else:
            self.game_over = self.shots_left == 0 and not self.shot_in_prog
            super().on_update(delta_time)

    def shoot(self, event=None):
        super().shoot(event=event)
        self.shots_left -= 1

    def on_draw(self):
        if not self.game_over:
            super().on_draw()
            self.shots_left_text.text = f"Shots Left: {self.shots_left}"
            self.shots_left_text.draw()
            self.score_text.text = f"Score: {self.score}"
            self.score_text.draw()
            return
        self.clear()
        self.game_over_text.text = f"Game Over!\nFinal Score: {self.score}"
        self.game_over_text.draw()

    def on_mouse_press(self, x, y, button, key_modifiers):
        if self.game_over:
            return
        super().on_mouse_press(x, y, button, key_modifiers)

    def handle_made_ball(self, ball):
        if ball.is_cue_ball():
            self.game_over = True
            self.shot_in_prog = False
            self.shots_left = 0
            self.score -= 1
        else:
            self.score += 1
        self.pool_ball_list.remove(ball)


class GymMiniGame(gym.Env, MiniGame):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, title="Pool", num_balls=1, shots=1, render_screen=True):
        gym.Env.__init__(self)
        MiniGame.__init__(self, width, height, title, num_balls, shots=shots)
        self.action_space = spaces.Box(low=np.array([-1, -1]).astype(np.float32),
                                       high=np.array([1, 1]).astype(np.float32), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros((self.num_balls+1, 2)).astype(np.float32),
                                            high=np.full((self.num_balls+1, 2),
                                                         np.array([SCREEN_WIDTH, SCREEN_HEIGHT])).astype(np.float32),
                                            shape=(self.num_balls+1, 2), dtype=np.float32)
        self.reward_range = (0, self.num_balls)
        self.render_screen = render_screen
        self.setup()


    def step(self, action: ActType, render_mode="human") -> Tuple[ObsType, float, bool, dict]:
        slope = action / np.linalg.norm(action)
        self.aim_slope = slope
        old_score = self.score
        self.shoot()
        while self.shot_in_prog:
            self.on_update(1 / FPS)
            if self.render_screen:
                self.render(render_mode)
        self.on_update(1 / FPS)

        observation = self.export_ball_positions()
        reward: int = self.score - old_score
        terminated: bool = self.game_over
        info = {}
        return observation, reward, terminated, info

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> \
            Union[ObsType, tuple[ObsType, dict]]:
        self.variable_setup()
        return self.export_ball_positions()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return arcade.get_image()
        elif mode == "human":
            self.dispatch_events()
            self.on_draw()
            self.flip()

    def close(self):
        print("closing window")
        MiniGame.close(self)
        gym.Env.close(self)



class BotMemory:
    def __init__(self, load_memory=False, capacity=50_000, save_rate=100):
        self.memory = []
        if load_memory:
            self.load_memory()
        self.capacity = capacity
        self.position = 0
        self.save_rate = save_rate

    def push(self, ball_positions, slope, score, done):
        mem = (ball_positions, slope, score, done)
        if self.position >= len(self.memory):
            self.memory.append(mem)
        else:
            self.memory[self.position] = mem

        self.position = (self.position + 1) % self.capacity

        if self.position % self.save_rate == 0:
            self.save_memory()

    def sample(self, size):
        return zip(*random.sample(self.memory, size))

    def save_memory(self):
        with open("SaveFiles/memory.pickle") as f:
            pickle.dump(self.memory, f)

    def load_memory(self):
        with open("SaveFiles/memory.pickle") as f:
            self.memory = pickle.load(f)

    def __len__(self):
        return len(self.memory)


class PoolBot:
    def __init__(self, memory):
        self.memory = memory

    def looping(self, env, episodes=1, speed_scaling=1):
        global FPS
        true_fps = FPS
        FPS /= speed_scaling
        for i in range(episodes):
            state = env.reset()
            env.render("human")
            while True:
                action = env.action_space.sample()
                new_state, reward, done, _ = env.step(action)
                self.memory.push(state, action, reward, done)
                if done:
                    break
                state = new_state
        FPS = true_fps
        print(len(self.memory))

    def tf_looping(self, env, episodes, speed_scaling=1):
        global FPS
        true_fps = FPS
        FPS /= speed_scaling
        for i in range(episodes):
            state = env.reset()
            env.render("human")
            while True:
                action = env.action_space.sample()
                new_state, reward, done, _ = env.step(action)
                self.memory.push(state, action, reward, done)
                if done:
                    break
                state = new_state
        FPS = true_fps
        print(len(self.memory))




def main():
    """ Main function """
    # Change the NUM_BALLS function at the top to choose a different number of balls
    # game = PoolGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, num_balls=NUM_BALLS)   # If you want to play unrestricted
    game = MiniGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, num_balls=NUM_BALLS, shots=5)
    game.setup()
    game.run()


if __name__ == "__main__":
    main()
