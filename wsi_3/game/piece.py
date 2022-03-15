from enum import Enum
from game.locals import SQUARE_SIZE


class PieceType(Enum):
    FOX = 1
    HOUND = -1


class Piece:
    def __init__(self, row, col, color, piece_type) -> None:
        self.row = row
        self.col = col
        self.color = color
        self.piece_type = piece_type
        self.x = 0
        self.y = 0
        self.calculate_pos()

    def calculate_pos(self):
        self.x = SQUARE_SIZE * self.col + SQUARE_SIZE // 2
        self.y = SQUARE_SIZE * self.row + SQUARE_SIZE // 2

    def move(self, row, col):
        self.row = row
        self.col = col
        self.calculate_pos()

    def possible_moves(self):
        moves = []
        if self.piece_type == PieceType.FOX:
            moves.append((self.row - 1, self.col - 1))
            moves.append((self.row - 1, self.col + 1))

        moves.append((self.row + 1, self.col - 1))
        moves.append((self.row + 1, self.col + 1))

        return moves

    def __repr__(self):
        return f"{self.piece_type}"
