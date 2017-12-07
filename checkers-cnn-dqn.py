import sys
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense, Convolution2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.models import Sequential

from checkers import CheckerBoard

from datetime import datetime
import time

from keras import backend as keras_backend


class DQNAgent:
    def __init__(self, state_depth, state_width, state_height, action_size, model_name, auto_save_time,
                 load_model_path=None,
                 epsilon_decay_steps=1000, test_mode=False):

        # get size of state and action
        self.state_depth = state_depth
        self.state_width = state_width
        self.state_height = state_height
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        if test_mode:
            self.epsilon = 0
            self.epsilon_min = 0
        else:
            self.epsilon = 1.0
            self.epsilon_min = 0.02

        self.memory_size = 100 * 100  # roughly 100-200 games of checkers

        self.batch_size = 256
        self.train_start = 4096

        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_decay_steps

        self.valid_actions = []
        self.model_saving_timer_start = 0

        self.auto_save_time = auto_save_time

        # create replay memory using deque
        self.memory = deque(maxlen=self.memory_size)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        self.model_save_dir = "models"
        self.model_name = model_name

        if load_model_path is not None:
            self.model.load_weights(load_model_path)

        print("DQN initialization: "
              "\n discount factor %f "
              "\n learning rate %f "
              "\n epsilon_min %f "
              "\n heat up steps %d "
              "\n epsilon_decay %f"
              "\n state_depth %f"
              "\n state_width %f"
              "\n state_height %f"
              % (self.discount_factor, self.learning_rate,
                 self.epsilon_min, self.train_start, self.epsilon_decay,
                 self.state_depth, self.state_width, self.state_height))

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()

        keras_backend.set_image_dim_ordering('th')

        model.add(Convolution2D(32, 2, 2, activation="relu",
                                input_shape=(self.state_depth, self.state_width, self.state_height)))

        model.add(Convolution2D(48, 2, 2, activation="relu"))

        model.add(Convolution2D(64, 2, 2, activation="relu"))

        model.add(Flatten())

        model.add(Dense(1024, activation='relu',
                        init='he_uniform'))

        model.add(Dense(1024, activation='relu',
                        init='he_uniform'))

        model.add(Dense(self.action_size, activation='linear',
                        init='he_uniform'))

        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

        if self.model_saving_timer_start == 0:
            self.model_saving_timer_start = time.time()
        else:
            current_time = time.time()
            if (current_time - self.model_saving_timer_start) > self.auto_save_time:
                print("saving model")
                now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
                self.model.save_weights("./" + self.model_save_dir + "/" + self.model_name + "-" + now + ".h5")
                self.model_saving_timer_start = 0

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        assert len(self.valid_actions) != 0, "valid actions should not be empty!"

        if np.random.rand() <= self.epsilon:
            return random.choice(self.valid_actions)
        else:
            state = np.array([state])  # num of examples, 1, self.channels, self.state_width, self.state_height

            q_value = self.model.predict(state)

            predicted_actions_values = q_value[0]

            predicted_action_value = None
            predicted_action_idx = None

            for idx, q_value in enumerate(predicted_actions_values):
                if idx in self.valid_actions and \
                        (predicted_action_value is None or q_value > predicted_action_value):
                    predicted_action_value = q_value
                    predicted_action_idx = idx

            return predicted_action_idx

    def set_valid_actions(self, valid_actions):
        self.valid_actions = valid_actions

    # save sample <s,a,r,s',s_rules,s'_rules> to the replay memory
    def append_sample(self, state, action, reward, next_state, done, valid_actions, next_valid_actions):
        self.memory.append((state, action, reward,
                            next_state, done,
                            valid_actions, next_valid_actions))

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_depth, self.state_width, self.state_height))
        update_target = np.zeros((batch_size, self.state_depth, self.state_width, self.state_height))
        action, reward, done = [], [], []
        valid_actions, next_valid_actions = [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])
            valid_actions.append(mini_batch[i][5])
            next_valid_actions.append(mini_batch[i][6])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            predicted_action_value = None

            # apply game rules
            predicted_actions_values = target_val[i]
            state_next_valid_actions = next_valid_actions[i]
            state_valid_actions = valid_actions[i]

            assert action[i] in state_valid_actions, "Action should be in valid actions!"

            for idx, q_value in enumerate(predicted_actions_values):
                # get highest q prediction
                if idx in state_next_valid_actions:
                    if predicted_action_value is None or q_value > predicted_action_value:
                        predicted_action_value = q_value

                # nullify all invalid states in target
                if not (idx in state_valid_actions):
                    target[i][idx] = 0

            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * predicted_action_value

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       nb_epoch=1, verbose=0)


class CheckersEnvironmentWrapper:
    def __init__(self):
        # env initialization
        self.actions = {}
        self.observation = []
        self.reward = 0
        self.done = False
        self.last_action_idx = 0

        # initialize the board
        self.board = CheckerBoard()

        self.depth = len(self.board.get_state_cwh())
        self.width = len(self.board.get_state_cwh()[0])
        self.height = len(self.board.get_state_cwh()[0][0])

        self.win_reward = 100
        self.defeat_reward = -100

        self.game_turns = 0
        self.score = 0

        self.enable_capturing_reward = False

        for idx, move in enumerate(self.board.get_all_moves()):
            self.actions[idx] = move

        print("total actions: ", len(self.actions))

        self.action_space_size = len(self.actions)

        self.reset()

    def update_game_info(self):
        self.observation = self.board.get_state_cwh()

    def restart_environment_episode(self):
        self.board = CheckerBoard()

        self.update_game_info()

        return self.observation

    def _idx_to_action(self, action_idx):
        return self.actions[action_idx]

    def get_valid_idx_actions(self):
        possible_idx_actions = []

        possible_moves = self.board.get_legal_moves()

        for idx, action in self.actions.items():
            if action in possible_moves:
                possible_idx_actions.append(idx)

        return possible_idx_actions

    def step(self, action_idx):
        assert self.board.get_current_player() == self.board.BLACK_PLAYER, "Training player should be black!"

        self.last_action_idx = action_idx

        action = self.actions[action_idx]

        # print("take action ", action_idx, " : ", action)

        white_pieces_before = self.board.get_white_num() + self.board.get_white_kings_num()
        white_kings_pieces_before = self.board.get_white_kings_num()
        black_pieces_before = self.board.get_black_num() + self.board.get_black_kings_num()
        black_kings_pieces_before = self.board.get_black_kings_num()

        self.board.make_move(action)

        if self.board.get_current_player() == self.board.WHITE_PLAYER:
            if not self.board.is_over():
                # make AI opponent move
                self.opponent_move()

        self.update_game_info()

        white_pieces = self.board.get_white_num() + self.board.get_white_kings_num()
        white_kings_pieces = self.board.get_white_kings_num()
        black_pieces = self.board.get_black_num() + self.board.get_black_kings_num()
        black_kings_pieces = self.board.get_black_kings_num()

        if self.board.is_over():
            print("black: p. %d, k. %d, white: p. %d, k. %d" % (
                black_pieces, black_kings_pieces, white_pieces, white_kings_pieces))

            if self.board.get_winner() == self.board.BLACK_PLAYER:
                # black wins
                print("black wins")
                self.reward = self.win_reward
            else:
                print("white wins")
                self.reward = self.defeat_reward
        else:
            if self.enable_capturing_reward:
                captured_whites = white_pieces_before - white_pieces
                captured_black = black_pieces_before - black_pieces

                self.reward = captured_whites - captured_black
            else:
                self.reward = 0

        self.score += self.reward
        self.game_turns += 1

        self.done = self.board.is_over()

        return self.observation, self.reward, self.done

    def opponent_move(self):
        current_player = self.board.get_current_player()

        moves = self.board.get_legal_moves()
        action = random.choice(moves)

        # print("opponent takes action ", action)

        self.board.make_move(action)

        if self.board.get_current_player() == current_player:
            # print("opponent takes a jump")
            self.opponent_move()

    def reset(self):
        self.restart_environment_episode()
        self.done = False
        self.reward = 0.0
        self.last_action_idx = 0

        self.game_turns = 0
        self.score = 0

        return self.observation, self.reward, self.done


def run_session(experiment_name="", model_name="checkers", test=False, model_path=None):
    env = CheckersEnvironmentWrapper()
    env.enable_capturing_reward = True

    auto_save_time = 30 * 60

    decay_steps = 200 * 200

    dqn_agent = DQNAgent(env.depth, env.width, env.height, env.action_space_size,
                         model_name, auto_save_time,
                         epsilon_decay_steps=decay_steps,
                         load_model_path=model_path,
                         test_mode=test)

    total_steps = 0

    episodes = 0

    game_memory_n = 200.0

    game_memory = deque(maxlen=int(game_memory_n))

    episode_desc = "training"
    if test:
        episode_desc = "testing"

    while True:
        print("------------------------------------------\n"
              "Start " + episode_desc + " episode\n"
                "------------------------------------------")

        done = False

        env.reset()

        episode_steps = 0

        while not done:
            # get action for the current state and go one step in environment
            state = env.observation

            valid_actions = env.get_valid_idx_actions()

            dqn_agent.set_valid_actions(valid_actions)

            action = dqn_agent.get_action(state)

            next_state, reward, done = env.step(action)

            next_valid_actions = env.get_valid_idx_actions()

            # save the sample <s, a, r, s'> to the replay memory
            dqn_agent.append_sample(state, action, reward,
                                    next_state, done,
                                    valid_actions, next_valid_actions)

            if not test:
                # every time step do the training
                dqn_agent.train_model()

            episode_steps += 1

            if done:
                if not test:
                    # every episode update the target model to be same with model
                    dqn_agent.update_target_model()

        total_steps += episode_steps

        game_memory.append(env.board.get_winner())

        black_victories = 0
        white_victories = 0

        for winner in game_memory:
            if winner == env.board.BLACK_PLAYER:
                # black wins
                black_victories += 1
            else:
                white_victories += 1

        b_win_percents = (black_victories/game_memory_n) * 100

        print("------------------------------------------\n" +
              experiment_name + ". Episode: %d. Steps: %d. "
                                "Total steps: %d. Total score: %d. Exploration rate: %f. "
                                "Black won: %f.\n"
                                "------------------------------------------"
              % (episodes, episode_steps, total_steps, env.score, dqn_agent.epsilon, b_win_percents))

        episodes += 1


if __name__ == "__main__":
    run_session(experiment_name="Conv, nullfication, large memory, capturing reward",
                model_name="checkers_conv3_2x1024_no_dropout")
