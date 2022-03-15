import random
from game.board import Board


def minimax(board: Board, depth, max_player):

    if depth == 0 or board.winner() != 0:
        # max_player = True <--- FOX
        return evaluate(board), 0  # [(0,0), (0,0)]

    moves = all_moves(board, max_player)
    random.shuffle(moves)

    if max_player:
        value = -100
        best_move = None
        for move in moves:
            board.move_piece(
                board.get_piece(move[0][0], move[0][1]), move[1][0], move[1][1]
            )
            temp_value = minimax(board, depth - 1, False)[0]
            # print(f"Temp: {temp_value}")
            if temp_value > value:
                value = temp_value
                best_move = move
            board.move_back(
                board.get_piece(move[1][0], move[1][1]), move[0][0], move[0][1]
            )
    else:
        value = 100
        best_move = None
        for move in moves:
            board.move_piece(
                board.get_piece(move[0][0], move[0][1]), move[1][0], move[1][1]
            )
            temp_value = minimax(board, depth - 1, True)[0]
            if temp_value < value:
                value = temp_value
                best_move = move
            board.move_back(
                board.get_piece(move[1][0], move[1][1]), move[0][0], move[0][1]
            )

    return value, best_move


def evaluate(board):
    # sprawdza czy jest winner
    # + TO WYGRANA FOXA/WILKA
    # - TO WYGRANA HOUNDÓW/OWIEC
    if board.winner() != 0:
        return board.winner()

    column_values = {0: -4, 1: -3, 2: -2, 3: -1, 4: 1, 5: 2, 6: 3, 7: 4}
    column_q = 0
    value = 0
    fox = board.get_pieces()[0]
    hounds = board.get_pieces()[1:]

    if fox_behind_hound(board):
        value += 0.2
    # if fox_touching_hound(board):
    #     value -= 0.4

    if all([h.row for h in hounds]):
        value -= 0.4

    # evaluate distance between fox and hounds
    value -= 1 / sum(distance(fox, h) for h in hounds)

    # evaluate fox row
    value += 0.1 / fox.row

    for h in hounds:
        column_q += column_values[h.col]

    # column distribution
    col_distribution = abs(column_q) * 0.05
    value += col_distribution
    # # print(col_distribution)

    minmax_condition = (max(h.row for h in hounds) - min(h.row for h in hounds)) * 0.05
    value += minmax_condition
    # print(minmax_condition)
    # value += (max(h.row for h in hounds) - min(h.row for h in hounds))*0.1
    # value += random.randint(-10, 10)*0.01

    return value


def distance(fox, hound):
    return (fox.row - hound.row) ** 2 + (fox.col - hound.col) ** 2
    # return ((fox.row - hound.row)**2 + (fox.col - hound.col)**2)**(1/2)


# można jednocześnie sprawdzać kilka warunków w forze
def fox_behind_hound(board: Board):
    behind = False
    fox = board.get_pieces()[0]
    hounds = board.get_pieces()[1:]
    for h in hounds:
        if fox.row < h.row:
            behind = True

    return behind


def fox_touching_hound(board):
    touching = False
    fox = board.get_pieces()[0]
    hounds = board.get_pieces()[1:]
    for h in hounds:
        if fox.row + 1 == h.row:
            if fox.col - 1 == h.col or fox.col + 1 == h.col:
                touching = True
    return touching


def all_moves(board: Board, max_player):
    moves = []

    # z którego na które pole
    # [(1, 6), (1, 8)]
    if max_player:
        moves = moves_of_singe_piece(board, board.get_pieces()[0])
        # print(f"m{moves}")
    else:
        pieces = board.get_pieces()[1:]
        for p in pieces:
            for m in moves_of_singe_piece(board, p):
                moves.append(m)
        # print(moves)
    return moves


def moves_of_singe_piece(board, piece):
    moves = []
    current_pos = (piece.row, piece.col)
    for move in board.get_valid_moves(piece):
        moves.append([current_pos, move])
    return moves
