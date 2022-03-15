"""
Introduction to Artificial Intelligence, Exercise 6:
Reinforcement lerning.
Author: Robert Kaczmarski 293377
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


VECTOR: list[float]
POSITION: tuple[int, int]
MAZE: list[VECTOR]
ACTIONS: list[POSITION]
VISITED_POINTS: list[POSITION]
Q_TABLE: dict[POSITION, VECTOR]
FIGURE = plt.Figure
AXIS = np.ndarray


def generate_maze(
    size: int,
    holes: float,
    start: POSITION,
    stop: POSITION,
    stop_value: float,
    hole_value: float,
    empty_value: float,
) -> MAZE:
    maze = [hole_value for _ in range(size * size)]
    maze = np.reshape(maze, (size, size)).astype(np.float32)

    for i in range(1, size - 1):
        for j in range(1, size - 1):
            random_value = np.random.random()
            if random_value >= holes:
                maze[j][i] = empty_value

    maze[start[1]][start[0]] = empty_value
    maze[stop[1]][stop[0]] = stop_value

    return maze


def validate_maze(
    maze: MAZE, start: POSITION, stop: POSITION, hole_value: float
) -> bool:
    """DFS"""
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    stack = []
    explored = []
    is_ok = False

    x = start[0]
    y = start[1]

    while True:
        for d in directions:
            temp_x = x + d[0]
            temp_y = y + d[1]
            if (
                temp_x >= 0
                and temp_y >= 0
                and temp_x < len(maze)
                and temp_y < len(maze)
            ):
                if maze[temp_y][temp_x] != hole_value:
                    if (temp_x, temp_y) not in explored:
                        stack.append((temp_x, temp_y))
        if not stack:
            break
        x, y = stack.pop()
        explored.append((x, y))
        if (x, y) == stop:
            is_ok = True
            break

    return is_ok


def display_maze(maze: MAZE) -> None:
    for m in maze:
        print(m)


def create_q_table(size: int, actions: ACTIONS) -> Q_TABLE:
    q_table = {}
    states = []
    for i in range(size):
        for j in range(size):
            states.append((i, j))

    for state in states:
        q_table[state] = [0.0 for _ in range(len(actions))]

    return q_table


def display_q_table(q_table: Q_TABLE) -> None:
    for k, v in q_table.items():
        print(k, v)


def is_end_state(maze: MAZE, current_pos: POSITION, empty_value: float) -> bool:
    x = current_pos[1]
    y = current_pos[0]

    if np.isclose(maze[y][x], empty_value):
        return False
    else:
        return True


# def get_next_action(q_table: Q_TABLE, current_pos: POSITION, epsilon: float) -> int:
#     value = np.random.uniform(0, 1)

#     if value < epsilon:
#         act_index = np.argmax(q_table[current_pos])
#         # act_index = np.random.randint(0, 4)
#     else:
#         # act_index = np.argmax(q_table[current_pos])
#         act_index = np.random.randint(0, 4)

#     return act_index


def get_new_pos(current_pos: POSITION, action: POSITION) -> POSITION:
    return (current_pos[0] + action[0], current_pos[1] + action[1])


def q_lerning(
    maze: MAZE,
    q_table: Q_TABLE,
    actions: ACTIONS,
    starting_point: POSITION,
    ending_point: POSITION,
    episodes: int,
    alfa: float,
    gamma: float,
    epsilon: float,
    empty_value: float,
    is_race: bool,
) -> VISITED_POINTS:
    f, ax = plt.subplots(1, 2)

    for episode in range(episodes):
        current_pos = starting_point
        random_current_pos = starting_point
        q_visited = []
        random_visited = [random_current_pos]
        random_in_game = True

        # continue taking actions (i.e., moving) until we reach a terminal state
        # (i.e., until we reach the item packaging area or crash into an item storage location)
        while not is_end_state(maze, current_pos, empty_value):

            # RANDOM CAR
            if random_in_game:
                random_action_index = np.random.randint(0, 4)
                random_current_pos = get_new_pos(
                    random_current_pos, actions[random_action_index]
                )
                if is_end_state(maze, random_current_pos, empty_value):
                    random_in_game = False
                else:
                    random_visited.append(random_current_pos)

            # Q_UBER
            action_index = np.argmax(q_table[current_pos])
            old_pos = current_pos
            q_visited.append(old_pos)
            current_pos = get_new_pos(current_pos, actions[action_index])

            reward = maze[current_pos[0]][current_pos[1]]
            old_q_value = q_table[old_pos][action_index]

            q_table[old_pos][action_index] = (1 - alfa) * old_q_value + alfa * (
                reward + gamma * np.max(q_table[current_pos])
            )

        if is_race:
            show_race(maze, q_visited, random_visited, episode, ax, f)
        else:
            show_quber(maze, q_visited, episode)

        if current_pos == ending_point:
            print(f"Episode: {episode} Q_uber car has finished!", end="  ")
        else:
            print(f"Episode: {episode} Q_uber car crashed!", end="  ")
        if random_current_pos == ending_point:
            print(f"Episode: {episode} Random car has finished!", end="  ")
        else:
            print(f"Episode: {episode} Random car crashed!", end="  ")
        print("")

    return q_visited


def show_race(
    maze: MAZE,
    visited: VISITED_POINTS,
    random_visited: VISITED_POINTS,
    episode: int,
    ax: AXIS,
    f: FIGURE,
) -> None:
    rows, cols = maze.shape
    q_uber = np.copy(maze)
    random_car = np.copy(maze)
    max_range = np.max([len(visited), len(random_visited)])

    for i in range(max_range):
        if i < len(visited):
            q_uber[visited[i][0]][visited[i][1]] = 15
            if i != 0 and i < len(visited):
                q_uber[visited[i - 1][0]][visited[i - 1][1]] = 10
        if i < len(random_visited):
            random_car[random_visited[i][0]][random_visited[i][1]] = 15
            if i != 0 and i < len(random_visited):
                random_car[random_visited[i - 1][0]][random_visited[i - 1][1]] = 10

        for j in range(0, 2):
            ax[j].grid("on")
            ax[j].set_xticks(np.arange(0.5, rows, 1))
            ax[j].set_yticks(np.arange(0.5, cols, 1))
            ax[j].set_xticklabels([])
            ax[j].set_yticklabels([])

        f.suptitle(f"Episode: {episode}")
        ax[0].title.set_text("Q-Uber")
        ax[0].imshow(q_uber, cmap="tab20b")
        ax[1].title.set_text("Random")
        ax[1].imshow(random_car, cmap="tab20b")
        plt.pause(0.001)

    ax[0].clear()
    ax[1].clear()


def show_quber(
    maze: MAZE, visited: VISITED_POINTS, episode: int, clear: bool = True
) -> None:
    rows, cols = maze.shape
    q_uber = np.copy(maze)

    for i, v in enumerate(visited):
        plt.grid("on")
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, rows, 1))
        ax.set_yticks(np.arange(0.5, cols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        q_uber[v[0]][v[1]] = 15
        if i != 0:
            q_uber[visited[i - 1][0]][visited[i - 1][1]] = 10
        plt.title(f"Episode: {episode}")
        plt.imshow(q_uber, cmap="tab20b")
        plt.pause(0.0001)
        if clear:
            plt.clf()


def show_maze(maze: MAZE, size: int, starting_point: POSITION):
    plt.grid("on")
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, size, 1))
    ax.set_yticks(np.arange(0.5, size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    maze[starting_point] = 5
    # im = plt.imshow(maze, cmap="tab20b")
    # values = np.unique(maze.ravel())
    # colors = [ im.cmap(im.norm(value)) for value in values]
    # labels = ["unpaid trainee", "empty", "start", "stop"]
    # patches = [ mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(values)) ]
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.imshow(maze, cmap="tab20b")
    plt.show()


def main() -> int:
    size = 10
    holes_ratio = 0.40
    starting_point = (1, 1)
    ending_point = (size - 2, size - 2)
    stop_value = 20
    hole_value = -20
    empty_value = -1

    maze = None
    random_seed = 1

    is_race = True
    episodes = 100
    # epsilon is not used
    epsilon = 0.9
    alfa = 0.8
    gamma = 0.8

    while True:
        np.random.seed(random_seed)
        temp_maze = generate_maze(
            size,
            holes_ratio,
            starting_point,
            ending_point,
            stop_value,
            hole_value,
            empty_value,
        )
        if validate_maze(temp_maze, starting_point, ending_point, hole_value):
            maze = temp_maze
            break
        else:
            random_seed += 1

    # display_maze(maze)
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    q_table = create_q_table(size, actions)

    # show_maze(maze, size, starting_point)
    visited_points = q_lerning(
        maze,
        q_table,
        actions,
        starting_point,
        ending_point,
        episodes,
        alfa,
        gamma,
        epsilon,
        empty_value,
        is_race,
    )

    # print(visited_points)
    # print(len(visited_points))
    show_quber(maze, visited_points, episodes, clear=False)
    plt.savefig("shortest.png")
    # show_maze(maze, size, starting_point)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
