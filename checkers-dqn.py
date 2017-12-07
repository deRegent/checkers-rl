import sys
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.models import Sequential

from checkers import CheckerBoard

from datetime import datetime
import time


# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def __init__(self, state_size, action_size, model_name, auto_save_time, load_model_path=None,
                 epsilon_decay_steps=1000, test_mode = False):
        # if you want to see Cartpole learning, then change to True
        self.render = False

        # get size of state and action
        self.state_size = state_size
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
        self.memory = deque(maxlen=10000)

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
        model.add(Dense(1024, input_dim=self.state_size, activation='relu',
                        init='he_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu',
                        init='he_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu',
                        init='he_uniform'))
        model.add(Dropout(0.1))
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

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
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

        self.width = len(self.board.get_state_vector())
        self.height = 1

        self.win_reward = 100
        self.defeat_reward = 0

        self.game_turns = 0
        self.score = 0

        for idx, move in enumerate(self.board.get_all_moves()):
            self.actions[idx] = move

        print("total actions: ", len(self.actions))

        self.action_space_size = len(self.actions)

        self.reset()

    def update_game_info(self):
        self.observation = self.board.get_state_vector()

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

def train_checkers():

    env = CheckersEnvironmentWrapper()

    state_size = env.width
    action_size = env.action_space_size

    model_name = "checkers_512_512_128"

    auto_save_time = 30 * 60

    decay_steps = 200 * 200

    dqn_agent = DQNAgent(state_size, action_size, model_name, auto_save_time, epsilon_decay_steps=decay_steps)

    total_steps = 0

    episodes = 0

    game_memory = deque(maxlen=200)

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
            state = np.reshape(state, [1, state_size])

            valid_actions = env.get_valid_idx_actions()

            dqn_agent.set_valid_actions(valid_actions)

            action = dqn_agent.get_action(state)

            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            next_valid_actions = env.get_valid_idx_actions()

            # save the sample <s, a, r, s'> to the replay memory
            dqn_agent.append_sample(state, action, reward,
                                    next_state, done,
                                    valid_actions, next_valid_actions)

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
              "Fully connected. Nullification. Episode: %d. Steps: %d. "
              "Total steps: %d. Total score: %d. Exploration rate: %f. "
              "Black won: %d. White won: %d\n"
              "------------------------------------------"
              % (episodes, episode_steps, total_steps, env.score, dqn_agent.epsilon, black_victories, white_victories))

        episodes += 1

def test_checkers():

    model_path = "models/checkers_512_512_128-2017-12-07-05-46-48.443508.h5"

    env = CheckersEnvironmentWrapper()

    state_size = env.width
    action_size = env.action_space_size

    model_name = "checkers_test"
    auto_save_time = 60 * 60

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

            next_valid_actions = env.get_valid_idx_actions()

            # save the sample <s, a, r, s'> to the replay memory
            dqn_agent.append_sample(state, action, reward,
                                    next_state, done,
                                    valid_actions, next_valid_actions)

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
              "Fully connected. Nullification. Episode: %d. Steps: %d. "
              "Total steps: %d. Total score: %d. Exploration rate: %f."
              "Black won: %d. White won: %d\n"
              "------------------------------------------"
              % (episodes, episode_steps, total_steps, env.score, dqn_agent.epsilon, black_victories, white_victories))

        episodes += 1

if __name__ == "__main__":

    train_checkers()

    # test_checkers()