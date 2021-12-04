import os
import sys

import numpy as np
import pickle

from tqdm import tqdm

TIC_TAC_TOE_ROWS = 50
TIC_TAC_TOE_COLS = 50


class GameState:
    def __init__(self, player1, player2):
        self.board = np.zeros((TIC_TAC_TOE_ROWS, TIC_TAC_TOE_COLS))
        self.available_position_set = set([(i, j) for i in range(0, 3) for j in range(0, 3)])
        self.p1 = player1
        self.p2 = player2
        self.isGameEnd = False
        self.boardHash = None
        # player 1 will make first move
        self.playerSymbol = 1

    # unique hash of current board state, in this case, it is a array
    def get_hash(self):
        self.boardHash = str(self.board.reshape(TIC_TAC_TOE_COLS * TIC_TAC_TOE_ROWS))
        return self.boardHash

    def winner(self):
        """
        In this function, we will decide who is the winner of the game.
        We may check the summation of rows, columns and diagonal respectively.
        :return: 1--player 1 win, -1--player 2 win.
        """
        # check summation of one row
        for i in range(TIC_TAC_TOE_ROWS):
            if sum(self.board[i, :]) == 5:
                self.isGameEnd = True
                return 1
            if sum(self.board[i, :]) == -5:
                self.isGameEnd = True
                return -1
        # check summation of one column
        for i in range(TIC_TAC_TOE_COLS):
            if sum(self.board[:, i]) == 5:
                self.isGameEnd = True
                return 1
            if sum(self.board[:, i]) == -5:
                self.isGameEnd = True
                return -1
        # check summation of one diagonal
        positive_diag_sum = sum([self.board[i, i] for i in range(TIC_TAC_TOE_COLS)])
        negative_diag_sum = sum(
            [self.board[i, TIC_TAC_TOE_COLS - i - 1] for i in range(TIC_TAC_TOE_COLS)])
        diag_sum = max(abs(positive_diag_sum), abs(negative_diag_sum))
        if diag_sum == 5:
            self.isGameEnd = True
            if positive_diag_sum == 5 or negative_diag_sum == 5:
                return 1
            else:
                return -1

        # Other situation: if there are no available positions on the board, the players will be tied.
        if len(self.available_position()) == 0:
            self.isGameEnd = True
            return 0
        # not end
        self.isGameEnd = False
        return None

    def available_position(self):
        """
        :return: available position tuples.
        """
        return list(self.available_position_set)

    def update_state(self, position):
        """
        :param position: positions are tuple, which denotes rows and columns in the chess board.
        :return: None
        """
        self.board[position] = self.playerSymbol
        self.available_position_set.remove(position)
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # We only know the results when the game ends. This is a Monte Carlo process.
    def give_reward(self):
        """
        Give rewards to each player in the training process.
        :return: None
        """
        result = self.winner()
        if result == 1:
            self.p1.give_reward(10)
            self.p2.give_reward(0)
        elif result == -1:
            self.p1.give_reward(0)
            self.p2.give_reward(10)
        elif result == 0:
            # We want the computer to be stronger. Thus tie is a bad case for the computers.
            self.p1.give_reward(3)
            self.p2.give_reward(7)
        else:
            pass

    def reset(self):
        """
        Clear the chess board and restart the game.
        :return:
        """
        self.board = np.zeros((TIC_TAC_TOE_ROWS, TIC_TAC_TOE_COLS))
        self.available_position_set = set([(i, j) for i in range(0, 3) for j in range(0, 3)])
        self.boardHash = None
        self.isGameEnd = False
        self.playerSymbol = 1

    def train(self, rounds=100):
        """
        Main training process, dump states to pickle
        :param rounds:
        :return: None
        """
        for _ in tqdm(range(rounds)):
            while not self.isGameEnd:
                positions = self.available_position()
                p1_action = self.p1.choose_action(positions, self.board, self.playerSymbol)
                self.update_state(p1_action)
                board_hash = self.get_hash()
                self.p1.add_state(board_hash)
                win = self.winner()
                if win is not None:
                    self.give_reward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break
                else:
                    positions = self.available_position()
                    p2_action = self.p2.choose_action(positions, self.board, self.playerSymbol)
                    self.update_state(p2_action)
                    board_hash = self.get_hash()
                    self.p2.add_state(board_hash)
                    win = self.winner()
                    if win is not None:
                        self.give_reward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
        self.p1.save_policy()
        self.p2.save_policy()

    def play_with_human(self, _who_is_first):
        """
        Interaction process with real player.
        :param _who_is_first: 0 denotes computer, 1 denotes humans.
        :return:
        """
        is_continue = True
        if _who_is_first == 0:
            while not self.isGameEnd and is_continue:
                positions = self.available_position()
                p1_action = self.p1.choose_action(positions, self.board, self.playerSymbol)
                self.update_state(p1_action)
                self.display_board()
                win = self.winner()
                if win is not None:
                    if win == 1:
                        print(self.p1.name, ", computer wins! You lose.")
                    else:
                        print("tie!")
                    self.reset()
                    break
                else:
                    # Player 2
                    positions = self.available_position()
                    p2_action = self.p2.choose_action(positions)

                    self.update_state(p2_action)
                    self.display_board()
                    win = self.winner()
                    if win is not None:
                        if win == -1:
                            print(self.p2.name, "wins! Congratulations! ")
                        else:
                            print("tie!")
                        self.reset()
                        break
            user_will = input("Want another try(y/n):")
            while user_will not in {'y', 'n', 'yes', 'no', 'Yes', 'No'}:
                user_will = input("Want another try(y/n):")
            if (user_will == 'n') or (user_will == 'no') or (user_will == 'No'):
                sys.exit(0)
        else:
            while not self.isGameEnd and is_continue:
                positions = self.available_position()
                p1_action = self.p1.choose_action(positions)
                self.update_state(p1_action)
                self.display_board()
                win = self.winner()
                if win is not None:
                    if win == 1:
                        print(self.p1.name, "wins! Congratulations! ")
                    else:
                        print("tie!")
                    self.reset()
                    break
                else:
                    positions = self.available_position()
                    p2_action = self.p2.choose_action(positions, self.board, self.playerSymbol)
                    self.update_state(p2_action)
                    self.display_board()
                    win = self.winner()
                    if win is not None:
                        if win == -1:
                            print(self.p2.name, "wins! You lose.")
                        else:
                            print("tie!")
                        self.reset()
                        break
            user_will = input("Want another try(y/n):")
            while user_will not in {'y', 'n', 'yes', 'no', 'Yes', 'No'}:
                user_will = input("Want another try(y/n):")
            if (user_will == 'n') or (user_will == 'no') or (user_will == 'No'):
                sys.exit(0)

    def display_board(self):
        """
        Use different character to mark two players. player 1: x  player 2: o
        :return: None
        """
        for i in range(0, TIC_TAC_TOE_ROWS):
            if i == 0:
                print('    1   2   3  ')
            print('  -------------')
            line = str(i + 1) + ' | '
            for j in range(0, TIC_TAC_TOE_COLS):
                character = ' '
                if self.board[i, j] == 1:
                    character = 'x'
                if self.board[i, j] == -1:
                    character = 'o'
                line += character + ' | '
            print(line)
        print('  -------------')


class Player:
    def __init__(self, name, exp_rate=0.3, alpha=0.2, decay_gamma=0.9):
        self.name = name
        self.states = []  # record all positions taken
        self.alpha = alpha
        self.exp_rate = exp_rate
        self.decay_gamma = decay_gamma
        self.states_value = {}

    def get_hash(self, board):
        """
        Get unique states on a board. There are 3^9 states in total.
        :param board:
        :return: None
        """
        boardHash = str(board.reshape(TIC_TAC_TOE_COLS * TIC_TAC_TOE_ROWS))
        return boardHash

    def choose_action(self, available_positions, current_board, symbol):
        """
        choose actions: randomly with probability exp_rate and select the maximum rewards with probability 1-exp_rate
        :param available_positions: available_positions
        :param current_board: current_board
        :param symbol: player mark
        :return: None
        """
        action = (-1, -1)
        # Sample from a uniform distribution and decide whether we should make move.
        if np.random.uniform(0, 1) <= self.exp_rate:
            idx = np.random.choice(len(available_positions))
            action = available_positions[idx]
        else:
            value_max = -999
            for p in available_positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.get_hash(next_board)
                value = 0 if self.states_value.get(
                    next_boardHash) is None else self.states_value.get(
                        next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    # append a hash state
    def add_state(self, _state):
        self.states.append(_state)

    def give_reward(self, reward):
        """
        At the end of game, backpropagation and update states value.
        Q[S,A] ← (1-α)*Q[S,A] + α*(R(S,a) + γ* max Q[S',a])
        :param reward: 0, 3, 7 and 10
        :return: None
        """
        for _st in reversed(self.states):
            if self.states_value.get(_st) is None:
                self.states_value[_st] = 0
            # R(s,a) = 0  because we do not know whether we will win in the game
            update_value = (1 - self.alpha) * self.states_value[_st] + self.alpha * (self.decay_gamma * reward)
            self.states_value[_st] = update_value
            reward = update_value

    def reset(self):
        self.states = []

    def save_policy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def load_policy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def choose_action(self, positions):
        """
        Interaction process on console
        :param positions: tuples like input.
        :return: None
        :exit: sys.exit(0)
        """
        while True:
            row = 0
            try:
                row = int(input("Input your action row:"))
            except ValueError:
                print('Wrong Input')
                pass
            while row < 1 or row > 3:
                print("Wrong Input Row. Input Range: 1-3")
                try:
                    row = int(input("Input your action row:"))
                except ValueError:
                    print('Wrong Input')
                    pass
            col = 0
            try:
                col = int(input("Input your action column:"))
            except ValueError:
                print('Wrong Input')
                pass
            while col < 1 or row > 3:
                print("Wrong Input Column. Input Range: 1-3")
                try:
                    col = int(input("Input your action column:"))
                except ValueError:
                    print('Wrong Input')
                    pass
            action = (row - 1, col - 1)
            if action in positions:
                return action

    def add_state(self, _state):
        pass

    def give_reward(self, reward):
        pass

    def reset(self):
        pass


if __name__ == '__main__':
    # training
    p1 = Player("p1")
    p2 = Player("p2")
    training_iterations = 1000000000
    if not os.path.isfile("five_p1") or not os.path.isfile("five_p2"):
        st = GameState(p1, p2)
        print("training...")
        st.train(training_iterations)

    computer1 = Player("computer", exp_rate=0)
    computer2 = Player("computer", exp_rate=0)
    computer1.load_policy("five_p1")
    computer2.load_policy("five_p2")
    while True:
        # play with human
        human_name = input("Who are you: ")
        print('Hello ' + human_name + ', welcome to the game! ')
        who_is_first = -1
        try:
            who_is_first = int(
                input("Who wanna be the first(0 for computer and 1 for human): "))
        except ValueError:
            print('Wrong Input')
            pass
        while who_is_first != 0 and who_is_first != 1:
            print("Wrong Input Row. Input Range: 0 or 1")
            try:
                who_is_first = int(
                    input(
                        "Who wanna be the first(0 for computer and 1 for human): ")
                )
            except ValueError:
                print('Wrong Input')
                pass
        computer = computer1 if who_is_first == 0 else computer2
        human = HumanPlayer(human_name)
        state = GameState(computer, human) if who_is_first == 0 else GameState(
            human, computer)
        state.play_with_human(who_is_first)
