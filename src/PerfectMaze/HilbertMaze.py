import json
import os
import pickle
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from rich import print

from .utils import _curve_print


class _HilbertCurve:
    def __init__(
            self,
            order: int,
            info: bool
    ):
        """
        Initializes the _HilbertCurve class.

        Args:
            order (int): The order of the Hilbert Curve.
            info (bool): Whether to display information about the curve during generation.
        """
        self.order = order
        self.info = info

    def _build(self, order) -> tuple:
        """
        Recursively constructs the Hilbert Curve of the given order and a map from coordinates to their index in the curve.

        Args:
            order (int): The order of the Hilbert Curve to build.

        Returns:
            tuple: A tuple containing two elements:
                - list[tuple[Any, Any]]: The list of coordinates representing the Hilbert Curve.
                - dict[tuple[Any, Any], int]: A dictionary mapping each coordinate in the curve to its index.
        """
        if order == 0:
            return [(0, 0)], {(0, 0): 0}
        # Generate order-1 order curves recursively
        lower_curve, lower_map = self._build(order - 1)
        self.size = 1 << (order - 1)  # 2^(order-1)
        curve = []
        coord_map = {}
        index = 0

        # The first quadrant: Rotation and flipping
        for x, y in lower_curve:
            new_x, new_y = y, x
            curve.append((new_x, new_y))
            coord_map[(new_x, new_y)] = index
            index += 1

        # The second quadrant: Translation
        for x, y in lower_curve:
            new_x, new_y = x, y + self.size
            curve.append((new_x, new_y))
            coord_map[(new_x, new_y)] = index
            index += 1

        # The third quadrant: Translation
        for x, y in lower_curve:
            new_x, new_y = x + self.size, y + self.size
            curve.append((new_x, new_y))
            coord_map[(new_x, new_y)] = index
            index += 1

        # The fourth quadrant: Rotation and flipping
        for x, y in lower_curve:
            new_x, new_y = 2 * self.size - 1 - y, self.size - 1 - x
            curve.append((new_x, new_y))
            coord_map[(new_x, new_y)] = index
            index += 1

        return curve, coord_map


class _HilbertMaze(_HilbertCurve):
    """"""

    def __init__(
            self,
            order: int,
            info: bool,
            save_dir: str = ".",
            save_name: str = "Hilbert_maze"
    ):
        """
        Initializes the _HilbertMaze class, inheriting from _HilbertCurve.

        Args:
            order     (int) : The order of the Hilbert Curve used to generate the maze.
            info     (bool) : Whether to display information about the maze during generation.
            save_dir  (str) : The directory to save the maze files. Defaults to ".".
            save_name (str) : The base name for the saved maze files. Defaults to "Hilbert_maze".
        """
        super().__init__(order, info)
        self.save_dir = save_dir
        self.save_name = save_name
        self.curve = None
        self.maze = None
        self.coord_map = None
        self.maze_start = self.maze_end = (-1, -1)
        self.solution_path = []  # 存储解路径坐标（按顺序）
        self.curve_maze_solve = []  # 存储解路径坐标（按坐标排序）

    # noinspection PyTypeChecker
    def _save_maze(self, path: str = None, filename: str = None):
        """
        Saves the maze, curve, and coordinate map to files.

        Args:
            path (str, optional):
                The directory to save the files. If None, uses self.save_dir.
                Defaults to None.
            filename (str, optional):
                The base filename for the curve file. If None, uses self.save_name + ".curve.txt".
                Defaults to None.
        """
        if path is None:
            path = self.save_dir
        if filename is None:
            filename = self.save_name + ".curve.txt"
        if self.maze is None:
            self._generate(self.order)
        with open(os.path.join(path, filename), "w", encoding="utf-8") as f:
            json.dump(self.curve, f)
        file2 = filename.split(".")[0] + ".coord_map"
        with open(os.path.join(path, file2), 'wb') as f:
            pickle.dump(self.coord_map, f)
        file3 = filename.split(".")[0] + ".maze"
        np.save(os.path.join(path, file3), self.maze)

    def _load_maze(self, maze_path, coord_map_path, curve_path):
        """
        Loads the maze, curve, and coordinate map from files.

        Args:
            maze_path      (str): The path to the saved maze file (.npy).
            coord_map_path (str): The path to the saved coordinate map file (.coord_map).
            curve_path     (str): The path to the saved curve file (.curve.txt).
        """
        self.maze = np.load(maze_path)
        with open(curve_path, 'r', encoding="utf-8") as f:
            self.curve = json.load(f)
        with open(coord_map_path, 'rb') as f:
            self.coord_map = pickle.load(f)
        self.maze_start = self.curve[0]
        self.maze_end = self.curve[-1]

    def _generate(self, order: int) -> tuple[Any, Any, Any]:
        """
        Generates the maze using the Hilbert Curve of the given order.

        Args:
            order (int): The order of the Hilbert Curve used to generate the maze.

        Returns:
            tuple: A tuple containing three elements:
                - numpy.ndarray: The generated maze grid (0 for paths, 1 for walls).
                - tuple[int, int]: The start coordinates of the maze (from the start of the curve).
                - tuple[int, int]: The end coordinates of the maze (from the end of the curve).
        """
        curve, coord_map = self._build(order)
        size = 1 << order  # 2^order
        total_cells = size * size

        if self.info:
            print(f"[INFO] Curve Start : {curve[0]}")
            print(f"[INFO] Curve End   : {curve[-1]}")
            print(f"[INFO] Curve Length: {len(curve)}")
            print(f"[INFO] Full Curve  : \n>>>\n")
            _curve_print(curve)
            print("\n<<<\n")
        self.curve = curve
        self.curve_start = curve[0]
        self.curve_end = curve[-1]
        self.coord_map = coord_map
        # The length of the list strictly follows the equation `f(x) = 4^x`

        # Create the maze grid (0= paths, 1= walls)
        # A `2*size+1` x `2*size+1` grid is used
        maze = np.ones((2 * size + 1, 2 * size + 1), dtype=int)

        # Mark all cell centers as pathways
        for i in range(size):
            for j in range(size):
                maze[2 * i + 1, 2 * j + 1] = 0

        # Save start and end coordinates (for entry/exit Settings)
        start_x, start_y = curve[0]
        end_x, end_y = curve[-1]

        # Connect adjacent cells (following Hilbert curve order)
        for idx, (x, y) in enumerate(curve):
            if idx == total_cells - 1:
                # The last cell has no further neighbors
                continue

            # Gets the position of the current cell in the maze grid
            forward_neighbors = []

            # Check the neighbors in all four directions
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    neighbor_idx = coord_map.get((nx, ny), -1)
                    # Consider only neighbors ahead of the curve (larger index)
                    if neighbor_idx > idx:
                        forward_neighbors.append((dx, dy, nx, ny))

            # Select a front neighbor at random and break through the wall
            if forward_neighbors:
                dx, dy, nx, ny = random.choice(forward_neighbors)
                # Calculate wall position (between current cell and neighbors)
                wall_x, wall_y = 2 * x + 1 + dx, 2 * y + 1 + dy
                maze[wall_x, wall_y] = 0  # Get through the wall

        # Break through the walls to set up entrances and exits
        # The exit is at the starting position (start of the curve)
        if start_y == 0:  # Left boundary
            maze[2 * start_x + 1, 0] = 0
        elif start_x == 0:  # Upper lateral boundary
            maze[0, 2 * start_y + 1] = 0

        # The entry is at the end (the end of the curve)
        if end_y == size - 1:  # Right boundary
            maze[2 * end_x + 1, 2 * size] = 0
        elif end_x == size - 1:  # Lower lateral boundary
            maze[2 * size, 2 * end_y + 1] = 0

        if self.info:
            print(f"[INFO] Maze Start: {curve[0]}")
            print(f"[INFO] Maze End  : {curve[-1]}")
            print(f"[INFO] Full Maze : \n>>>\n{maze}\n<<<\n")
        self.maze = maze
        self.maze_start = curve[0]
        self.maze_end = curve[-1]
        return self.maze, self.maze_start, self.maze_end

    def _show(self, solve: bool = False, overwrite: bool = False) -> None:
        """
        Visualizes the maze using matplotlib. Optionally displays the solution path.

        Args:
            solve     (bool):
                Whether to solve the maze and display the solution path. Defaults to False.
            overwrite (bool):
                Whether to regenerate the maze and solution before displaying. Defaults to False.
        """
        # Create a custom color mapping
        if self.maze is not None and not overwrite:
            maze = self.maze
            start = self.maze_start
            end = self.maze_end
        else:
            maze, start, end = self._generate(self.order)
        if solve and not self.solution_path or overwrite:
            self._solve()

        cmap = ListedColormap([
            'white',
            'black',
            '#E49B0F',  # start
            '#89cff0',  # end
            '#93C572'  # solve
        ])

        # Copy the maze for visual marking
        display_maze = maze.copy().astype(float)

        if solve:
            for (x, y) in self.solution_path:
                display_maze[x, y] = 4

        s_x, s_y = 2 * start[0] + 1, 2 * start[1] + 1
        e_x, e_y = 2 * end[0] + 1, 2 * end[1] + 1
        display_maze[s_x, s_y] = 2
        display_maze[e_x, e_y] = 3

        if start[1] == 0:
            display_maze[s_x, 0] = 2
        elif start[0] == 0:
            display_maze[0, s_y] = 2

        if end[1] == (maze.shape[1] - 1) // 2 - 1:
            display_maze[e_x, -1] = 3
        elif end[0] == (maze.shape[0] - 1) // 2 - 1:
            display_maze[-1, e_y] = 3

        plt.figure(figsize=(10, 10))
        plt.imshow(display_maze, cmap=cmap, interpolation='nearest')
        for i in range(0, maze.shape[0] + 1):
            plt.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(0, maze.shape[1] + 1):
            plt.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        title = f"Hilbert Maze Solution\norder = {self.order}" if solve else f"Hilbert Maze\norder = {self.order}"
        plt.title(title)
        plt.axis('off')
        plt.show()

    def _solve(self):
        """
        Solves the maze using a depth-first search (DFS) algorithm.

        Finds the path from the start to the end of the maze and stores it in self.solution_path and self.curve_maze_solve.

        Raises:
            RuntimeError: If no solution is found (should not occur for valid mazes).
        """
        if self.maze is None:
            self._generate(self.order)

        # get maze's size
        size = 1 << self.order
        total_size = 2 * size + 1

        # the location of start and end in the maze
        start = (2 * self.maze_start[0] + 1, 2 * self.maze_start[1] + 1)
        end = (2 * self.maze_end[0] + 1, 2 * self.maze_end[1] + 1)

        # solve
        stack = [start]
        visited = np.zeros((total_size, total_size), dtype=bool)
        parent = {}
        visited[start] = True

        while stack:
            x, y = stack.pop()

            if (x, y) == end:
                path = []
                current = end
                while current != start:
                    path.append(current)
                    current = parent[current]
                path.append(start)
                path.reverse()
                self.solution_path = path
                curve_path = []
                for px, py in path:
                    if px % 2 == 1 and py % 2 == 1:
                        curve_path.append(((px - 1) // 2, (py - 1) // 2))
                self.curve_maze_solve = sorted(curve_path, key=lambda p: (p[0], p[1]))
                return

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < total_size and 0 <= ny < total_size and not visited[nx, ny]:
                    if self.maze[nx, ny] == 0 or (nx, ny) == end:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
                        parent[(nx, ny)] = (x, y)

        raise RuntimeError("[Err ] Maze No Solution Error! This is should not occurred.")


class HilbertMaze(_HilbertMaze):
    VALID_INT_LENGTH = [8, 16, 32, 64]

    def __init__(
            self,
            order: int = 3,
            full_maze: bool = False,
            int_length: int = 16,
            info: bool = False,
            save_dir: str = ".",
            save_name: str = "Hilbert_maze"
    ):
        f"""
        Initializes the HilbertMaze class, inheriting from _HilbertMaze.
        
        Validates input parameters and sets up options for full maze printing.
        
        Args:
            order      (int): 
                The order of the Hilbert Curve. Defaults to 3.
            full_maze (bool): 
                Whether to print complete maze information. Defaults to False.
            int_length (int): 
                The integer type used for printing full maze information. 
                Must be in [8, 16, 32, 64]. Defaults to 16.
            info      (bool): 
                Whether to display information during generation. Defaults to False.
            save_dir   (str): 
                Directory to save maze files. Defaults to ".".
            save_name  (str): 
                Base name for saved maze files. Defaults to "Hilbert_maze".
        
        Raises:
            ValueError: If order is not a positive integer or int_length is not in the valid list.
        """

        try:
            order = int(order)
        except ValueError as e:
            raise ValueError(
                f"The `HilbertMaze.__init__`'s arg `order` " +
                f"must be a value that can be converted to an integer"
            ) from e
        if order <= 0:
            raise ValueError(
                f"The `HilbertMaze.__init__`'s arg `order` " +
                f"must greater than zero"
            )
        if int_length not in self.VALID_INT_LENGTH:
            raise ValueError(
                f"The `HilbertMaze.__init__`'s arg `int_length` " +
                f"can only be one of {self.VALID_INT_LENGTH}"
            )
        super().__init__(order, info, save_dir, save_name)
        self.full_maze = full_maze
        self.int_length = int_length
        if self.full_maze:
            np.set_printoptions(threshold=np.iinfo(getattr(np, f"int{self.int_length}")).max, linewidth=1000)

        self.maze = None

    def generate(self) -> tuple[Any, Any, Any]:
        """
        Generates the maze using the Hilbert Curve of the specified order.

        Returns:
            tuple: A tuple containing the maze grid, start coordinates, and end coordinates (same as _generate).
        """
        return self._generate(self.order)

    def display(self, overwrite: bool = False):
        """
        Displays the maze visualization.

        Args:
            overwrite (bool):
                Whether to regenerate the maze before displaying. Defaults to False.
        """
        self._show(overwrite=overwrite)

    def save_maze(self, **kwargs):
        """
        Saves the maze, curve, and coordinate map to files.
        Accepts keyword arguments for path and filename.

        Args:
            **kwargs: Keyword arguments passed to _save_maze, including path and filename.
        """
        self._save_maze(**kwargs)

    def load_maze(self, *args):
        """
        Loads the maze, curve, and coordinate map from files.

        Args:
            *args: Arguments passed to _load_maze, including maze_path, coord_map_path, and curve_path.
        """
        self._load_maze(*args)

    def solve_and_display(self, overwrite: bool = False):
        """
        Solves the maze and displays the visualization with the solution path.

        Args:
            overwrite (bool):
            Whether to regenerate and resolve the maze before displaying. Defaults to False.
        """
        self._show(solve=True, overwrite=overwrite)
