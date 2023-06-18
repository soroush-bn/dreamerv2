################################################################################################################
# Authors:                                                                                                     #
# Soroush Baghernezhad                                                                              #
################################################################################################################
import numpy as np


#####################################################################################################################
# Env
# todo  write description
#####################################################################################################################
# todo bug in calculating reward (what to do with edges ??)
class Env:
    def __init__(self, ramping=None):
        self.channels = {
            'paddle_player': 0,
            'paddle_computer': 1,
            'ball': 2,
        }
        self.action_map = ['n', 'l', 'u', 'r', 'd', 'f']
        self.random = np.random.RandomState()
        self.reset()

    # Update environment according to agent action
    def act(self, a):
        r = 0
        if (self.terminal):
            return r, self.terminal

        a = self.action_map[a]

        # Resolve player action
        if (a == 'r'):
            self.paddle_player_x = max(1, self.paddle_player_x - 1)
        elif (a == 'l'):
            self.paddle_player_x = min(8, self.paddle_player_x + 1)

        # computer movement
        if (self.paddle_computer_x == 8 and self.paddle_computer_dx == 1) or (
                self.paddle_computer_x == 1 and self.paddle_computer_dx == -1):
            self.paddle_computer_dx *= -1
        self.paddle_computer_x += self.paddle_computer_dx

        # Update ball position
        # hardcoded todo need fix
        if (self.ball_x == 0 and self.ball_dx < 0): self.ball_dx = +1
        if (self.ball_x == 9 and self.ball_dx > 0): self.ball_dx = -1
        if (self.ball_y == 0 and self.ball_dy < 0): self.ball_dy = +1
        if (self.ball_y == 9 and self.ball_dy > 0): self.ball_dy = -1
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        #  check for collision
        if self.ball_x == 0 or self.ball_x == 9:
            self.ball_dx *= -1

        # paddle collision with ball
        if self.ball_y == 1:
            if self.ball_x in range(self.paddle_computer_x - 1, self.paddle_computer_x + 2):
                if self.ball_x == self.paddle_computer_x:
                    self.ball_dy = 1

                elif self.ball_x == self.paddle_computer_x - 1:
                    self.ball_dy = 1
                    self.ball_dx = -1

                elif self.ball_x == self.paddle_computer_x + 1:
                    self.ball_dy = 1
                    self.ball_dx = 1
            else:
                self.goal_player += 1
                r += 1
        elif self.ball_y == 8:
            if self.ball_x in range(self.paddle_player_x - 1, self.paddle_player_x + 2):
                if self.ball_x == self.paddle_player_x:
                    self.ball_dy = -1
                elif self.ball_x == self.paddle_player_x - 1:
                    self.ball_dy = -1
                    self.ball_dx = -1
                elif self.ball_x == self.paddle_player_x + 1:
                    self.ball_dy = -1
                    self.ball_dx = 1
            else:
                self.goal_computer += 1

        if self.goal_computer == 21 or self.goal_player == 21:
            self.terminal = True

        return r, self.terminal

    # Query the current level of the difficulty ramp, difficulty does not ramp in this game, so return None
    def difficulty_ramp(self):
        return None

    # Process the game-state into the 10x10xn state provided to the agent and return
    def state(self):
        state = np.zeros((10, 10, len(self.channels)), dtype=bool)
        state[self.ball_x, self.ball_y, self.channels['ball']] = 1
        state[self.paddle_player_x - 1:self.paddle_player_x + 2, 9, self.channels['paddle_player']] = 1
        state[self.paddle_computer_x - 1:self.paddle_computer_x + 2, 0, self.channels['paddle_computer']] = 1
        return state

    # Reset to start state for new episode
    def reset(self):

        self.terminal = False

        self.ball_x = 4
        self.ball_y = 4
        self.ball_dx = +1
        self.ball_dy = +1
        # refer to x_center from the 3pixel of the paddle
        # ranging between 1 to 8
        self.paddle_player_x = 5
        self.paddle_computer_x = 5
        self.paddle_computer_dx = +1
        self.goal_player = False
        self.goal_computer = False

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10, 10, len(self.channels)]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ['n', 'l', 'r']
        return [self.action_map.index(x) for x in minimal_actions]
