from termcolor import colored
import random
from math import copysign

STARTING_WHITES = [(1, 0), (3, 0), (5, 0), (7, 0),
                   (0, 1), (2, 1), (4, 1), (6, 1),
                   (1, 2), (3, 2), (5, 2), (7, 2)]
STARTING_BLACKS = [(0, 5), (2, 5), (4, 5), (6, 5),
                   (1, 6), (3, 6), (5, 6), (7, 6),
                   (0, 7), (2, 7), (4, 7), (6, 7)]

BOARD_SIZE = 8


def is_even(num):
    return num % 2 == 0


def is_game_piece(x, y):
    return (is_even(y) and (not is_even(x))) or ((not is_even(y)) and is_even(x))


class CheckerBoard:
    def __init__(self):
        self.BLACK_PLAYER, self.WHITE_PLAYER = 0, 1
        self.EMPTY, self.BLACK, self.BLACK_KING, self.WHITE, self.WHITE_KING = 0, -1, -3, 1, 3

        self.state = {}
        self.state_array = [x[:] for x in [[self.EMPTY] * BOARD_SIZE] * BOARD_SIZE]

        for y, row in enumerate(self.state_array):
            for x, piece in enumerate(row):
                self.state[(x, y)] = self.EMPTY

        for position in STARTING_BLACKS:
            self.state[position] = self.BLACK

        for position in STARTING_WHITES:
            self.state[position] = self.WHITE

        self.current_player = self.BLACK_PLAYER
        self.game_over = False

        self.simplified = False

    def get_state_cwh(self):
        return [self.get_state_matrix()]

    def get_state_matrix(self):
        for y, row in enumerate(self.state_array):
            for x, piece in enumerate(row):
                self.state_array[y][x] = self.state[(x, y)]

        return self.state_array

    def get_state_vector(self):
        state_vector = []

        for y, row in enumerate(self.get_state_matrix()):
            for x, piece in enumerate(row):
                if is_game_piece(x, y):
                    piece_vector = [0] * 4
                    # 0 for black, 1 for black king, 2 for white, 3 for white king
                    if piece == self.BLACK:
                        piece_vector[0] = 1
                    elif piece == self.BLACK_KING:
                        piece_vector[1] = 1
                    elif piece == self.WHITE:
                        piece_vector[2] = 1
                    elif piece == self.WHITE_KING:
                        piece_vector[3] = 1
                    state_vector.extend(piece_vector)

        return state_vector

    def is_over(self):
        return len(self.get_legal_moves()) == 0

    def get_winner(self):
        if not self.is_over():
            return None

        if self.current_player == self.BLACK_PLAYER:
            return self.WHITE_PLAYER
        else:
            return self.BLACK_PLAYER

    def get_white_num(self):
        num = 0
        for position, piece in self.state.items():
            if self.get_ownage(position) == self.WHITE_PLAYER:
                num += 1
        return num

    def get_black_num(self):
        num = 0
        for position, piece in self.state.items():
            if self.get_ownage(position) == self.BLACK_PLAYER:
                num += 1
        return num

    def get_white_kings_num(self):
        num = 0
        for position, piece in self.state.items():
            if self.get_ownage(position) == self.WHITE_PLAYER and piece == self.WHITE_KING:
                num += 1
        return num

    def get_black_kings_num(self):
        num = 0
        for position, piece in self.state.items():
            if self.get_ownage(position) == self.BLACK_PLAYER and piece == self.BLACK_KING:
                num += 1
        return num

    def get_current_player_name(self):
        return self.get_player_name(self.current_player)

    def get_player_name(self, player):
        if player == self.BLACK_PLAYER:
            return "BLACK"
        else:
            return "WHITE"

    def get_legal_moves(self):
        capturing_moves = []
        regular_moves = []

        for position, piece in self.state.items():
            if self.get_ownage(position) == self.current_player:
                capturing_moves_from = self.get_capturing_moves_from(position)
                if len(capturing_moves_from) > 0:
                    capturing_moves.extend(capturing_moves_from)

                regular_moves_from = self.get_regular_moves_from(position)
                if len(regular_moves_from) > 0:
                    regular_moves.extend(regular_moves_from)

        if len(capturing_moves) > 0:
            return capturing_moves
        else:
            return regular_moves

    def get_all_moves(self):
        moves = []

        for position, piece in self.state.items():
            if is_game_piece(position[0], position[1]):
                moves.extend(self.get_all_moves_from(position))

        return moves

    def get_current_player(self):
        return self.current_player

    def make_move(self, move):
        assert len(move) == 2 or len(move[0]) == 2 or len(move[1]) == 2, "Invalid move format"

        start_position = move[0]
        end_position = move[1]

        move_x = end_position[0] - start_position[0]
        move_y = end_position[1] - start_position[1]

        if abs(move_x) > 1 and abs(move_y) > 1:
            # capturing move detected
            captured_piece_x = start_position[0] + copysign(1, move_x)
            captured_piece_y = start_position[1] + copysign(1, move_y)

            captured_position = (captured_piece_x, captured_piece_y)

            capturing_ownage = self.get_ownage(captured_position)

            assert not (capturing_ownage is None or capturing_ownage == self.current_player), "Invalid capturing move"

            self.state[captured_position] = self.EMPTY

        assert self.state[end_position] == self.EMPTY, "Invalid move: end position is not empty"

        self.state[end_position] = self.state[start_position]
        self.state[start_position] = self.EMPTY

        if self.get_ownage(end_position) == self.BLACK_PLAYER and end_position[1] == 0:
            self.state[end_position] = self.BLACK_KING
        elif self.get_ownage(end_position) == self.WHITE_PLAYER and end_position[1] == BOARD_SIZE - 1:
            self.state[end_position] = self.WHITE_KING

        if not self.is_jump_required():
            if self.current_player == self.BLACK_PLAYER:
                self.current_player = self.WHITE_PLAYER
            else:
                self.current_player = self.BLACK_PLAYER

    def __str__(self):
        str = ""
        for y, row in enumerate(self.get_state_matrix()):
            for x, piece in enumerate(row):
                if self.simplified:
                    id = " "
                    man = "-"
                    king = "+"
                else:
                    id = " (%d,%d)E " % (x, y)
                    man = " (%d,%d)M " % (x, y)
                    king = " (%d,%d)K " % (x, y)

                if piece == self.BLACK:
                    id = colored(man, 'red')
                elif piece == self.BLACK_KING:
                    id = colored(king, 'red')
                elif piece == self.WHITE:
                    id = colored(man, 'green')
                elif piece == self.WHITE_KING:
                    id = colored(king, 'green')
                str = str + id
            str = str + "\n"
        return str

    # private

    def get_all_moves_from(self, position):
        end_positions = [
            (position[0] - 1, position[1] - 1),
            (position[0] - 1, position[1] + 1),
            (position[0] + 1, position[1] - 1),
            (position[0] + 1, position[1] + 1),
            (position[0] - 2, position[1] - 2),
            (position[0] - 2, position[1] + 2),
            (position[0] + 2, position[1] - 2),
            (position[0] + 2, position[1] + 2),
        ]

        moves = []

        for end_position in end_positions:
            if 0 <= end_position[0] < BOARD_SIZE and 0 <= end_position[1] < BOARD_SIZE:
                moves.append([position, end_position])

        return moves

    def get_regular_moves_from(self, position):
        piece = self.state[position]

        end_positions = []
        if piece == self.BLACK:
            end_positions = [
                (position[0] - 1, position[1] - 1),
                (position[0] + 1, position[1] - 1),
            ]
        elif piece == self.WHITE:
            end_positions = [
                (position[0] - 1, position[1] + 1),
                (position[0] + 1, position[1] + 1),
            ]
        else:
            end_positions = [
                (position[0] - 1, position[1] - 1),
                (position[0] - 1, position[1] + 1),
                (position[0] + 1, position[1] - 1),
                (position[0] + 1, position[1] + 1),
            ]

        moves = []

        for end_position in end_positions:
            if 0 <= end_position[0] < BOARD_SIZE and 0 <= end_position[1] < BOARD_SIZE \
                    and self.get_ownage(end_position) is None:
                moves.append([position, end_position])

        return moves

    def get_capturing_moves_from(self, position):

        piece = self.state[position]

        end_positions = []
        if piece == self.BLACK:
            end_positions = [
                (position[0] - 2, position[1] - 2),
                (position[0] + 2, position[1] - 2),
            ]
        elif piece == self.WHITE:
            end_positions = [
                (position[0] - 2, position[1] + 2),
                (position[0] + 2, position[1] + 2),
            ]
        else:
            end_positions = [
                (position[0] - 2, position[1] - 2),
                (position[0] - 2, position[1] + 2),
                (position[0] + 2, position[1] - 2),
                (position[0] + 2, position[1] + 2),
            ]

        moves = []

        for end_position in end_positions:
            if 0 <= end_position[0] < BOARD_SIZE and 0 <= end_position[1] < BOARD_SIZE:
                if self.get_ownage(end_position) is None:
                    move_x = end_position[0] - position[0]
                    move_y = end_position[1] - position[1]

                    captured_piece_x = position[0] + copysign(1, move_x)
                    captured_piece_y = position[1] + copysign(1, move_y)

                    captured_position = (captured_piece_x, captured_piece_y)
                    capturing_ownage = self.get_ownage(captured_position)

                    if (capturing_ownage is not None) and capturing_ownage != self.current_player:
                        moves.append([position, end_position])

        return moves

    def is_jump_required(self):
        for position, piece in self.state.items():
            if self.get_ownage(position) == self.current_player and len(self.get_capturing_moves_from(position)) > 0:
                return True
        return False

    def get_ownage(self, position):
        piece = self.state[position]
        if piece == self.BLACK or piece == self.BLACK_KING:
            return self.BLACK_PLAYER
        elif piece == self.WHITE or piece == self.WHITE_KING:
            return self.WHITE_PLAYER
        else:
            return None


# test random game
if __name__ == "__main__":
    board = CheckerBoard()

    print("-------------------SANITY CHECK-------------------")

    print(board)

    print("state matrix")
    print(board.get_state_matrix())

    print("state vector")
    print(board.get_state_vector())

    print("legal moves")
    print(board.get_legal_moves())

    print("all moves")
    print(board.get_all_moves())
    print("all moves num ", len(board.get_all_moves()))

    print("-------------------STARTING RANDOM GAME-------------------")

    while not board.is_over():
        print(board)

        moves = board.get_legal_moves()
        action = random.choice(moves)

        print("Player %s takes an action: " % board.get_current_player_name(), action)

        board.make_move(action)

    print(board)

    print("The winner is: ", board.get_player_name(board.get_winner()))
