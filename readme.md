# PerfectMaze

[English](https://github.com/starwindv/PerfectMaze/blob/main/readme.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/starwindv/PerfectMaze/blob/main/LICENSE)
![Python Version](https://img.shields.io/badge/python-3.10+-orange)

PerfectMaze is a Python library for generating perfect mazes, which are mazes without loops and where all paths are connected. The current version implements a maze generation algorithm based on the Hilbert Curve.

## Main Features

- 🧩 Maze generation based on the Hilbert Curve
- 📊 Maze visualization (using Matplotlib)
- 💾 Maze save and load functionality
- 🧭 Maze solving algorithm (DFS)
- ✅ Perfect maze verification feature

## Installation

```bash
pip install PerfectMaze
```

## Quick Start

```python
from PerfectMaze.HilbertMaze import HilbertMaze

# Create a Hilbert maze (order 3)
maze_gen = HilbertMaze(order=3)

# Generate the maze
maze, start, end = maze_gen.generate()

# Display the maze
maze_gen.display()

# Solve and display the maze path
maze_gen.solve_and_display()
```

## Example Output

![Hilbert Maze](https://github.com/StarWindv/PerfectMaze/blob/main/assets/order6_1.png)
*Example of a Hilbert maze with order 6*

## API Reference

### `HilbertMaze` Class

#### Initialization Parameters:
```python
def __init__(
    order: int = 3,
    full_maze: bool = False,
    int_length: int = 16,
    info: bool = False,
    save_dir: str = ".",
    save_name: str = "Hilbert_maze"
): ...
```

#### Main Methods:
- `generate()`: Generate the maze, returns (maze_grid, start_coord, end_coord)
- `display(overwrite=False)`: Visualize the maze
- `solve_and_display(overwrite=False)`: Solve and visualize the maze path
- `save_maze(**kwargs)`: Save the maze to a file
- `load_maze(*args)`: Load the maze from a file

### Utility Functions
```python
from PerfectMaze import is_perfect_maze
maze_grid = ...
# Verify if the maze is a perfect maze
is_perfect = is_perfect_maze(maze_grid)
```

## Dependencies

- Python 3.10+
- rich
- numpy
- matplotlib

## Contribution

Contributions are welcome! Please submit issues or suggestions through GitHub Issues and code improvements through Pull Requests.

## License

This project is licensed under the [MIT](https://github.com/starwindv/PerfectMaze/blob/main/LICENSE) License.