import numpy as np
from collections import deque


def _curve_print(curve: list):
    """
    Prints a list in a formatted multi-column layout with aligned columns.

    The output is formatted as a 2D grid with up to 8 columns per row. Column widths
    are dynamically adjusted based on the maximum element width in each column.

    Args:
        curve: The input list to be printed. Elements can be of any printable type.

    Example output:
        [
            (0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (3, 0), (3, 1), (2, 1),
            (2, 2), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2), (0, 2), (0, 3),
            (0, 4), (1, 4), (1, 5), (0, 5), (0, 6), (0, 7), (1, 7), (1, 6),
            (2, 6), (2, 7), (3, 7), (3, 6), (3, 5), (2, 5), (2, 4), (3, 4),
            (4, 4), (5, 4), (5, 5), (4, 5), (4, 6), (4, 7), (5, 7), (5, 6),
            (6, 6), (6, 7), (7, 7), (7, 6), (7, 5), (6, 5), (6, 4), (7, 4),
            (7, 3), (7, 2), (6, 2), (6, 3), (5, 3), (4, 3), (4, 2), (5, 2),
            (5, 1), (4, 1), (4, 0), (5, 0), (6, 0), (6, 1), (7, 1), (7, 0),
        ]

    Note:
        This function prints directly to stdout and returns nothing.
    """
    if not curve:
        print("[]")
        return

    # Calculate maximum column widths for alignment
    col_widths = [0] * 8
    for i, point in enumerate(curve):
        col_idx = i % 8
        s = str(point)
        if len(s) > col_widths[col_idx]:
            col_widths[col_idx] = len(s)

    total = len(curve)
    lines = total // 8
    remainder = total % 8

    # Print in formatted multi-column layout
    print("[")
    for i in range(lines):
        start_idx = i * 8
        line_elements = []
        for j in range(8):
            element = curve[start_idx + j]
            s = str(element)
            line_elements.append(s.ljust(col_widths[j]))
        line_str = "    " + ", ".join(line_elements)
        line_str += ","
        print(line_str)

    # Handle remaining elements
    if remainder > 0:
        start_idx = lines * 8
        line_elements = []
        for j in range(remainder):
            element = curve[start_idx + j]
            s = str(element)
            line_elements.append(s.ljust(col_widths[j]))
        line_str = "    " + ", ".join(line_elements)
        print(line_str + ",")
    print("]")


def is_perfect_maze(maze: np.ndarray) -> bool:
    """
    Determines if a given maze meets the criteria of a 'perfect maze'.

    A perfect maze must satisfy two conditions:
    1. All path cells (value 0) are connected (single connected component)
    2. Contains no cycles (exactly (n-1) edges for n path cells)

    Args:
        maze: A 2D numpy array representing the maze where:
              0 = Path cell (traversable)
              Non-zero = Wall/blocked cell

    Returns:
        True if the maze is a perfect maze, False otherwise.

    Examples:
        >>> is_perfect_maze(np.array([[0, 1], [0, 0]]))
        True
        >>> is_perfect_maze(np.array([[0, 0], [0, 0]]))  # Has cycle
        False
    """
    rows, cols = maze.shape

    # Identify all path cells (value 0)
    path_cells = np.argwhere(maze == 0)
    if len(path_cells) == 0:
        return False
    num_paths = len(path_cells)

    # Single cell case
    if num_paths == 1:
        return True

    # BFS traversal to check connectivity and count edges
    visited = set()
    edge_count = 0
    start = tuple(path_cells[0])
    queue = deque([start])
    visited.add(start)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 4-connectivity

    while queue:
        current = queue.popleft()
        current_row, current_col = current

        for dr, dc in directions:
            neighbor_row = current_row + dr
            neighbor_col = current_col + dc

            # Check boundaries
            if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                neighbor = (neighbor_row, neighbor_col)

                if maze[neighbor_row, neighbor_col] == 0:
                    # Count edges using ordered pairs to avoid duplication
                    if current < neighbor:
                        edge_count += 1

                    # Process unvisited neighbors
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

    # Verify both conditions for perfect maze
    return len(visited) == num_paths and edge_count == num_paths - 1
