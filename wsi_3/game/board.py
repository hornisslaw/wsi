from game.locals import COLS, YELLOW, GREEN
from game.piece import Piece, PieceType


class Board:
    def __init__(self, fox_starting_col) -> None:
        self.pieces = []
        self.create_board(fox_starting_col)

    def create_board(self, fox_starting_col):
        # create fox
        if fox_starting_col % 2 != 0 or fox_starting_col < 0 or fox_starting_col > 7:
            print(f"Starting col {fox_starting_col} is wrong, changing to 0")
            fox_starting_col = 0
        self.pieces.append(Piece(7, fox_starting_col, YELLOW, PieceType.FOX))
        # create hounds
        for c in range(1, COLS, 2):
            self.pieces.append(Piece(0, c, GREEN, PieceType.HOUND))

    def get_pieces(self):
        return self.pieces

    def get_valid_moves(self, piece: Piece):
        valid_moves = []
        for move in piece.possible_moves():
            if self.is_move_valid(piece, move[0], move[1]):
                valid_moves.append(move)
        return valid_moves

    def get_piece(self, row, col):
        piece = None
        for p in self.pieces:
            if p.row == row and p.col == col:
                piece = p
        # print(f"get_piece: {piece}")
        return piece

    def is_move_valid(self, piece: Piece, row, col) -> bool:
        # Check if the new position is the same as old position
        if piece.row == row and piece.col == col:
            return False
        # Check if new positions is outside the board
        elif row < 0 or row > 7 or col < 0 or col > 7:
            return False
        # Check if Hound movind backwards
        elif piece.piece_type is PieceType.HOUND and piece.row >= row:
            return False
        # Check if move is on diagonal tile row
        elif piece.row + 1 != row and piece.row - 1 != row:
            return False
        # Check if move is on diagonal tile col
        elif piece.col + 1 != col and piece.col - 1 != col:
            return False

        valid_move = True
        # Check if there is already a piece
        for p in self.pieces:
            if row == p.row and col == p.col:
                valid_move = False

        return valid_move

    def move_piece(self, piece: Piece, row, col):
        if piece:
            valid = self.is_move_valid(piece, row, col)
            if valid:
                piece.row, piece.col = row, col
                piece.move(row, col)
            else:
                print("wrong move")
        else:
            print(f"Cant move {piece}")

    def move_back(self, piece, row, col):
        if piece:
            piece.row, piece.col = row, col
            piece.move(row, col)
        else:
            print(f"Cant move {piece}")

    def winner(self):
        # fox has made its way to row 0
        value = 10
        if self.pieces[0].row == 0:
            return value
        # fox cant move
        elif len(self.get_valid_moves(self.pieces[0])) == 0:
            return -value
        # hound cant move
        elif all(
            len(m) == 0
            for m in [self.get_valid_moves(self.pieces[i]) for i in range(1, 5)]
        ):
            return value
        else:
            return 0
