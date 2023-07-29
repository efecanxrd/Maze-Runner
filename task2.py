from heapq import heappop, heappush
from typing import List, Tuple
import time
import tkinter as tk
import numpy as np

class AStar:
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)

    def search(self, start: Tuple[int, int], end: Tuple[int, int]):
        open_list = []
        closed_list = set()
        heappush(open_list, (0, start, []))

        while open_list:
            current_cost, current_node, current_path = heappop(open_list)
            if current_node == end:
                return current_path + [end]
            if current_node in closed_list:
                continue
            closed_list.add(current_node)

            for neighbor in self.get_neighbors(current_node):
                neighbor_cost = current_cost + 1
                if neighbor in closed_list:
                    continue
                if neighbor not in (node for _, node, _ in open_list):
                    heappush(open_list, (neighbor_cost + self.get_heuristic(neighbor, end), neighbor, current_path + [current_node]))
                else:
                    for cost, node, path in open_list:
                        if node == neighbor and cost > neighbor_cost + self.get_heuristic(neighbor, end):
                            open_list.remove((cost, node, path))
                            heappush(open_list, (neighbor_cost + self.get_heuristic(neighbor, end), neighbor, current_path + [current_node]))
                            break

    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = node
        neighbors = []
        if x > 0 and not self.grid[x-1][y]:
            neighbors.append((x-1, y))
        if x < self.width-1 and not self.grid[x+1][y]:
            neighbors.append((x+1, y))
        if y > 0 and not self.grid[x][y-1]:
            neighbors.append((x, y-1))
        if y < self.height-1 and not self.grid[x][y+1]:
            neighbors.append((x, y+1))
        return neighbors

    def get_heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])


import random

def create_maze(size: int) -> List[List[int]]:
    grid = [[0 for _ in range(size)] for _ in range(size)]

    # Randomly place walls in the grid
    for i in range(size):
        for j in range(size):
            if random.random() < 0.2:
                grid[i][j] = 1

    # Choose start and end points
    start = (random.randint(0, size-1), random.randint(0, size-1))
    while grid[start[0]][start[1]] == 1:
        start = (random.randint(0, size-1), random.randint(0, size-1))
    end = (random.randint(0, size-1), random.randint(0, size-1))
    while grid[end[0]][end[1]] == 1 or end == start:
        end = (random.randint(0, size-1), random.randint(0, size-1))

    # Set start and end points in the grid
    grid[start[0]][start[1]] = 2
    grid[end[0]][end[1]] = 3


from random import randint
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib

class Maze:
    def __init__(self, size):
        self.size = size
        self.grid = [[0 for i in range(size)] for j in range(size)]
        self.start = None
        self.end = None
        self.generate_maze()

    def generate_maze(self):
        self.start = (0,0)
        self.end =  (self.size-1, self.size-1)
        while self.start == self.end:
            self.end = (randint(0, self.size-1), randint(0, self.size-1))

        for i in range(self.size*self.size//3):
            x, y = randint(0, self.size-1), randint(0, self.size-1)
            if (x, y) == self.start or (x, y) == self.end:
                continue
            self.grid[x][y] = 1

    def solve_maze(self):
        solved = False
        while not solved:
            a_star = AStar(deepcopy(self.grid))
            path = a_star.search(self.start, self.end)
            if not path:
                break
            for i in range(1, len(path)):
                self.grid[path[i][0]][path[i][1]] = 2
            solved = True

    def plot_maze(self):
        colors = ['white', 'black', 'black', 'red']
        cmap = matplotlib.colors.ListedColormap(colors)
        plt.figure(figsize=(6,6))
        plt.pcolor(self.grid[::-1], cmap=cmap, edgecolors='k', linewidths=2)
        plt.xticks(np.arange(0, self.size, 1)), plt.yticks(np.arange(0, self.size, 1))
        plt.show()
grid_size = input('Grid size: ')
def create():
    global maze
    maze = Maze(int(grid_size))
create()
def calistir():
    global maze
    start_time = time.time()
    maze.solve_maze()
    end_time = time.time()
    elapsed_time_ms = int((end_time - start_time) * 1000)
    print(f'Elapsed time: {elapsed_time_ms}ms')
    maze.plot_maze()

if __name__ == '__main__':
    root = tk.Tk()
    ana_cerceve = tk.Frame(root)
    ana_cerceve.pack()

    calistir_butonu = tk.Button(ana_cerceve, text="Çalıştır", command=calistir)
    calistir_butonu.pack(side="left")

    root.title("App")
    root.geometry("200x100")
    root.mainloop()