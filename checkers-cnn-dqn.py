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

# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def __init__(self, state_depth, state_width, state_height, action_size, model_name, auto_save_time, load_model_path=None,
                 epsilon_decay_steps=1000, test_mode = False):

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
            self.epsilon_min = 0.05

        self.batch_size = 128
        self.train_start = 1000

        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_decay_steps

        self.valid_actions = []
        self.model_saving_timer_start = 0

        self.auto_save_time = auto_save_time

        # create replay memory using deque
        self.memory = deque(maxlen=2000)

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
              % (self.discount_factor, self.learning_rate, self.epsilon_min, self.train_start, self.epsilon_decay))

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()

        keras_backend.set_image_dim_ordering('th')

        model.add(Convolution2D(32, 3, 3, activation="relu", input_shape=(self.state_depth, self.state_width, self.state_height)))
        model.add(Convolution2D(48, 3, 3, activation="relu"))
        model.add(Convolution2D(64, 3, 3, activation="relu"))

        model.add(Flatten())

        model.add(Dense(1024, activation='relu',
                        init='he_uniform'))
        model.add(Dropout(0.2))

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
        has_valid_actions = len(self.valid_actions) != 0

        if has_valid_actions:
            if np.random.rand() <= self.epsilon:
                return random.choice(self.valid_actions)
            else:

                state_cnn = [state] # 1, self.channels, self.state_width, self.state_height

                q_value = self.model.predict(state)

                predicted_actions_values = q_value[0]

                predicted_action_value = None
                predicted_action_idx = None

                for idx, q_value in enumerate(predicted_actions_values):
                    if idx in self.valid_actions and (
                            predicted_action_value is None or q_value > predicted_action_value):
                        predicted_action_value = q_value
                        predicted_action_idx = idx

                return predicted_action_idx
        else:
            print("don't have valid actions!")
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            else:
                q_value = self.model.predict(state)
                return np.argmax(q_value[0])

    def set_valid_actions(self, valid_actions):
        self.valid_actions = valid_actions

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
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

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

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

        for idx, move in enumerate(self.board.get_all_moves()):
            self.actions[idx] = move

        print("total actions: ", len(self.actions))

        self.action_space_size = len(self.actions)

        self.reset()

    def is_white_piece(self, idx):
        piece_vec = self.observation[idx * 4:idx * 4 + 4]
        return piece_vec[1] == 1 or piece_vec[3] == 1

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
            captured_whites = white_pieces_before - white_pieces
            captured_black = black_pieces_before - black_pieces

            self.reward = captured_whites - captured_black

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

def train_checkers():

    env = CheckersEnvironmentWrapper()

    model_name = "checkers_cnn"

    auto_save_time = 30 * 60

    decay_steps = 200 * 200

    dqn_agent = DQNAgent(env.depth, env.width, env.height, env.action_space_size,
                         model_name, auto_save_time, epsilon_decay_steps=decay_steps)

    total_steps = 0

    episodes = 0

    game_memory = deque(maxlen=1000)

    while True:
        print("------------------------------------------\n"
              "Start episode\n"
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

            # save the sample <s, a, r, s'> to the replay memory
            dqn_agent.append_sample(state, action, reward, next_state, done)

            # every time step do the training
            dqn_agent.train_model()

            episode_steps += 1

            if done:
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

        print("------------------------------------------\n"
              "Episode was finished. Episode: %d. Steps: %d. "
              "Total steps: %d. Total score: %d. Exploration rate: %f."
              "Last 1000 games. Black won: %d. White won: %d\n"
              "------------------------------------------"
              % (episodes, episode_steps, total_steps, env.score, dqn_agent.epsilon, black_victories, white_victories))

        episodes += 1

def test_checkers():

    model_path = "models/checkers_512_512_128_drop-2017-12-04-04-44-31.692527.h5"

    env = CheckersEnvironmentWrapper()

    state_size = env.width
    action_size = env.action_space_size

    model_name = "checkers_train"
    auto_save_time = 30 * 60

    decay_steps = 200 * 200

    dqn_agent = DQNAgent(state_size, action_size, model_name, auto_save_time,
                         epsilon_decay_steps=decay_steps,
                         load_model_path = model_path,
                         test_mode=True)

    total_steps = 0

    episodes = 0

    max_episodes = 10000
    black_victories = 0
    white_victories = 0

    while episodes < max_episodes:
        print("------------------------------------------\n"
              "Start test episode\n"
              "------------------------------------------")

        done = False

        env.reset()

        episode_steps = 0

        while not done:

            # get action for the current state and go one step in environment
            state = env.observation
            state = np.reshape(state, [1, state_size])

            valid_actions = env.get_valid_idx_actions()

            dqn_agent.set_valid_actions(valid_actions)

            action = dqn_agent.get_action(state)

            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # save the sample <s, a, r, s'> to the replay memory
            dqn_agent.append_sample(state, action, reward, next_state, done)

            # every time step do the training
            # dqn_agent.train_model()

            episode_steps += 1

            if done:
                # every episode update the target model to be same with model
                # dqn_agent.update_target_model()

                if env.board.get_winner() == env.board.BLACK_PLAYER:
                    # black wins
                    black_victories += 1
                else:
                    white_victories += 1


        total_steps += episode_steps

        print("------------------------------------------\n"
              "Episode was finished. Episode: %d. Steps: %d. "
              "Total steps: %d. Total score: %d. Exploration rate: %f."
              "Black won: %d. White won: %d\n"
              "------------------------------------------"
              % (episodes, episode_steps, total_steps, env.score, dqn_agent.epsilon, black_victories, white_victories))

        episodes += 1

if __name__ == "__main__":

    train_checkers()

    # test_checkers()