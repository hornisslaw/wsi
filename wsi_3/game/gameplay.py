import pygame
import random

from game.board import Board
from game.locals import ROWS, COLS, BLACK, RED, YELLOW, GREEN, WHITE, SQUARE_SIZE
from game.piece import PieceType
from minimax import minimax


class Gameplay:
    def __init__(self, window, fox_starting_pos) -> None:
        self.selected = None
        self.fox_starting_pos = fox_starting_pos
        self.board = Board(self.fox_starting_pos)
        self.turn = PieceType.FOX  # zmienić na fox/hound
        self.window = window
        self.valid_moves = []

    def update(self):
        self.draw()
        self.draw_valid_moves(self.valid_moves)
        pygame.display.update()

    def draw(self):
        self.draw_squares()
        self.draw_pieces()

    def draw_squares(self):
        self.window.fill(BLACK)
        for row in range(ROWS):
            for col in range(row % 2, COLS, 2):
                square = (
                    row * SQUARE_SIZE,
                    col * SQUARE_SIZE,
                    SQUARE_SIZE,
                    SQUARE_SIZE,
                )
                pygame.draw.rect(self.window, RED, square)

    def draw_pieces(self):
        radius = SQUARE_SIZE // 3
        for p in self.board.get_pieces():
            pygame.draw.circle(self.window, p.color, (p.x, p.y), radius)

    def draw_valid_moves(self, moves):
        for move in moves:
            row, col = move
            circle = (
                col * SQUARE_SIZE + SQUARE_SIZE // 2,
                row * SQUARE_SIZE + SQUARE_SIZE // 2,
            )
            pygame.draw.circle(self.window, WHITE, circle, SQUARE_SIZE // 5)

    def select(self, row, col):
        piece = self.board.get_piece(row, col)
        if piece and (piece.piece_type is self.turn):
            self.selected = piece
            self.valid_moves = self.board.get_valid_moves(piece)
            print(self.valid_moves)
            return True

        return False

    def move_selected_piece(self, row, col):

        if self.selected:
            piece = self.board.get_piece(row, col)
            if self.selected is not piece:
                self.selected = piece
            else:
                self.moved_piece(row, col)

            # if not piece_moved:
            #     self.selected = None
            # else:
            #     self.change_turn()

    def moved_piece(self, row, col):
        piece = self.board.get_piece(row, col)
        if piece is None and (row, col) in self.valid_moves:
            # if self.selected and piece is None and (row, col) in self.valid_moves:
            self.board.move_piece(self.selected, row, col)
            self.selected = None
            self.change_turn()
            self.valid_moves = []

    def change_turn(self):
        if self.turn is PieceType.FOX:
            self.turn = PieceType.HOUND
        else:
            self.turn = PieceType.FOX

    def auto_move(self, start_row, start_col, end_row, end_col):
        self.board.move_piece(
            self.board.get_piece(start_row, start_col), end_row, end_col
        )

    def random_move(self):
        piece = None
        moves = []
        random_move = None
        if self.turn is PieceType.FOX:
            piece = self.board.get_pieces()[0]
            moves = self.board.get_valid_moves(piece)
        else:
            pieces = self.board.get_pieces()[1:]
            while True:
                piece = random.choice(pieces)
                moves = self.board.get_valid_moves(piece)
                if moves:
                    break
        random_move = random.choice(moves)
        self.board.move_piece(piece, random_move[0], random_move[1])

    def get_board(self):
        return self.board

    def win(self):
        return self.board.winner()

    def __repr__(self):
        return f"Turn: {self.turn}"

    def game_loop(self, clock, FPS, depth, i):
        run = True
        total_moves = 0
        while run:
            clock.tick(FPS)

            if self.turn is PieceType.FOX:
                value, best = minimax(self.get_board(), depth, True)
                # print(f"Value: {value}")
                self.auto_move(best[0][0], best[0][1], best[1][0], best[1][1])
                # print(f"Best move Fox is from {best[0]} to {best[1]} with value of {value}")
                # self.random_move()
            else:
                # value, best = minimax(self.get_board(), depth, False)
                # print(f"Best move Hounds is from {best[0]} to {best[1]} with value of {value}")
                # gameplay.auto_move(best[0][0], best[0][1], best[1][0], best[1][1])
                # self.auto_move(best[0][0], best[0][1], best[1][0], best[1][1])
                self.random_move()
            total_moves += 1
            self.change_turn()

            if self.win() != 0:
                if self.win() == 10:
                    # print("Fox won")
                    winner = "fox"
                else:
                    # print("Hounds won")
                    winner = "hounds"
                self.save_winner_to_file(
                    winner, depth, self.fox_starting_pos, total_moves, i
                )
                # print(f"{gameplay.win()} won!")
                run = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    row, col = self.get_mouse_pos(pos)
                    if self.selected:
                        print(f"{self.selected.col}")
                        self.move_selected_piece(row, col)
                    else:
                        self.select(row, col)

                # if event.type == pygame.KEYDOWN:
                #     if event.key == pygame.K_SPACE:
                #         if self.turn is PieceType.FOX:
                #             value, best = minimax(self.get_board(), depth, True)
                #             print(f"Minimax value: {value}")
                #             print(f"Best move for fox is {best[1]}")
                #             self.auto_move(best[0][0], best[0][1], best[1][0], best[1][1])
                #             # self.random_move()
                #         else:
                #             value, best = minimax(self.get_board(), depth, False)
                #             print(f"Minimax value: {value}")
                #             print(f"Best move Hounds is from {best[0]} to {best[1]} with value of {value}")
                #             self.auto_move(best[0][0], best[0][1], best[1][0], best[1][1])
                #             # self.random_move()
                #         total_moves += 1
                #         self.change_turn()
            self.update()

    pygame.quit()

    def get_mouse_pos(pos: tuple):
        x, y = pos
        row = y // SQUARE_SIZE
        col = x // SQUARE_SIZE
        return row, col

    def save_winner_to_file(
        self, winner: str, depth: int, fsc: int, total_fox_moves: int, i
    ):
        with open("results.txt", "a") as f:
            f.write(f"\n{i}. {depth}:{fsc}:{winner}:{total_fox_moves}")
