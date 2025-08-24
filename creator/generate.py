"""
Note:
功能：
1. 加载预训练的生成器模型
2. 生成指定阶数的迷宫
3. 验证迷宫完美性
4. 多种输出格式（文本、图像、JSON）
5. 批量生成和单次生成模式

使用方法：
$ python generate.py --order 3 --num 5 --output_dir mazes
"""

import os
import argparse
import json
import numpy as np
import torch
from PIL import Image
from MazeTrain import Config, Generator
from PerfectMaze import is_perfect_maze


config = Config()


class MazeGenerator:
    """迷宫生成器，使用预训练的GAN模型生成完美迷宫"""

    def __init__(self, checkpoint_path):
        """初始化迷宫生成器

        Args:
            checkpoint_path (str): 生成器模型检查点路径
        """
        # 初始化生成器模型
        self.generator = Generator(
            noise_dim=config.NOISE_DIM,
            embed_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            max_order=config.MAX_ORDER
        )

        # 加载模型权重
        self.load_checkpoint(checkpoint_path)
        self.generator.eval()

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)

    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点

        Args:
            checkpoint_path (str): 检查点文件路径
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if 'generator_state' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator_state'])
            print(f"Loaded generator weights from {checkpoint_path}")
        elif 'state_dict' in checkpoint:
            # 兼容不同保存格式
            self.generator.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded generator weights from {checkpoint_path}")
        else:
            raise ValueError("Checkpoint file does not contain generator weights")

    def generate(self, order, num_mazes=1, threshold=0.5, validate=True):
        """生成指定阶数的迷宫

        Args:
            order (int): 迷宫阶数 (0-7)
            num_mazes (int, optional): 生成数量. Defaults to 1.
            threshold (float, optional): 二值化阈值. Defaults to 0.5.
            validate (bool, optional): 是否验证迷宫完美性. Defaults to True.

        Returns:
            list: 生成的迷宫列表，每个元素为字典:
                {
                    'maze': np.ndarray,  # 迷宫矩阵 (0=通路, 1=墙壁)
                    'is_perfect': bool,   # 是否完美迷宫
                    'order': int,         # 迷宫阶数
                    'size': int           # 迷宫尺寸 (边长)
                }
        """
        if not (0 <= order <= config.MAX_ORDER):
            raise ValueError(f"Order must be between 0 and {config.MAX_ORDER}")

        with torch.no_grad():
            # 准备输入
            noise = torch.randn(num_mazes, config.NOISE_DIM, device=self.device)
            orders = torch.full((num_mazes,), order, dtype=torch.long, device=self.device)

            # 生成迷宫
            generated = self.generator(noise, orders)

            # 转换为numpy数组
            generated_np = generated.cpu().numpy()

            # 处理结果
            results = []
            for i in range(num_mazes):
                print(f"Processed the : {i+1}")
                # 二值化
                binary_maze = (generated_np[i] > threshold).astype(np.int32)

                # 验证迷宫完美性
                is_perfect = is_perfect_maze(binary_maze) if validate else False
                # print(is_perfect)

                # 获取迷宫尺寸
                size = binary_maze.shape[0]

                results.append({
                    'maze': binary_maze,
                    'is_perfect': is_perfect,
                    'order': order,
                    'size': size
                })

        return results

    @staticmethod
    def save_as_text(maze, file_path):
        """将迷宫保存为文本文件

        Args:
            maze (np.ndarray): 迷宫矩阵
            file_path (str): 输出文件路径
        """
        np.savetxt(file_path, maze, fmt="%d")

    @staticmethod
    def save_as_image(maze, file_path, cell_size=10, wall_color=(0, 0, 0), path_color=(255, 255, 255)):
        """将迷宫保存为图片

        Args:
            maze (np.ndarray): 迷宫矩阵
            file_path (str): 输出文件路径
            cell_size (int, optional): 每个单元格的像素大小. Defaults to 10.
            wall_color (tuple, optional): 墙壁颜色 (RGB). Defaults to (0, 0, 0).
            path_color (tuple, optional): 路径颜色 (RGB). Defaults to (255, 255, 255).
        """
        # 创建图像
        height, width = maze.shape
        img = Image.new("RGB", (width * cell_size, height * cell_size), wall_color)
        pixels = img.load()

        # 绘制迷宫
        for y in range(height):
            for x in range(width):
                if maze[y, x] == 0:  # 通路
                    for i in range(cell_size):
                        for j in range(cell_size):
                            px = x * cell_size + i
                            py = y * cell_size + j
                            pixels[px, py] = path_color

        img.save(file_path)

    @staticmethod
    def save_as_json(maze_data, file_path):
        """将迷宫数据保存为JSON文件

        Args:
            maze_data (dict): 迷宫数据
            file_path (str): 输出文件路径
        """
        # 转换numpy数组为列表
        data_to_save = maze_data.copy()
        data_to_save['maze'] = data_to_save['maze'].tolist()

        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)

    @staticmethod
    def print_maze(maze, start_char='S', end_char='E', path_char=' ', wall_char='#'):
        """在控制台打印迷宫

        Args:
            maze (np.ndarray): 迷宫矩阵
            start_char (str, optional): 起点字符. Defaults to 'S'.
            end_char (str, optional): 终点字符. Defaults to 'E'.
            path_char (str, optional): 路径字符. Defaults to ' '.
            wall_char (str, optional): 墙壁字符. Defaults to '#'.
        """
        # 寻找起点和终点（简单实现，实际应用中应该由算法确定）
        height, width = maze.shape

        # 寻找入口（通常在边缘）
        start_pos = None
        for i in range(height):
            if maze[i, 0] == 0:
                start_pos = (i, 0)
                break
        if start_pos is None:
            for j in range(width):
                if maze[0, j] == 0:
                    start_pos = (0, j)
                    break

        # 寻找出口（通常在对面边缘）
        end_pos = None
        for i in range(height):
            if maze[i, width - 1] == 0:
                end_pos = (i, width - 1)
                break
        if end_pos is None:
            for j in range(width):
                if maze[height - 1, j] == 0:
                    end_pos = (height - 1, j)
                    break

        # 打印迷宫
        for i in range(height):
            row = []
            for j in range(width):
                if (i, j) == start_pos:
                    char = start_char
                elif (i, j) == end_pos:
                    char = end_char
                else:
                    char = path_char if maze[i, j] == 0 else wall_char
                row.append(char)
            print(''.join(row))


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Generate Maze')
    parser.add_argument('--order', type=int, required=True,
                        help='Maze Order (0-7)')
    parser.add_argument('--num', type=int, default=1,
                        help='The number which you want generate (默认: 1)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_less.pth',
                        help='Weights Path (Default: checkpoints/checkpoint_epoch_less.pth)')
    parser.add_argument('--output_dir', type=str, default='generated_mazes',
                        help='Output Dir (Default: generated_mazes)')
    parser.add_argument('--format', type=str, default='all', choices=['text', 'image', 'json', 'all', 'no'],
                        help='Output Type (Default: all)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值 (默认: 0.5)')
    parser.add_argument('--validate', action='store_true',
                        help='验证生成的迷宫是否为完美迷宫')
    parser.add_argument('--print', action='store_true',
                        help='在控制台打印迷宫')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化迷宫生成器
    try:
        generator = MazeGenerator(args.checkpoint)
    except Exception as e:
        print(f"初始化迷宫生成器失败: {e}")
        return

    # 生成迷宫
    try:
        mazes = generator.generate(
            order=args.order,
            num_mazes=args.num,
            threshold=args.threshold,
            validate=args.validate
        )
    except Exception as e:
        print(f"生成迷宫失败: {e}")
        return

    # 保存和输出结果
    perfect_count = 0
    if args.format != "no":
        for i, maze_data in enumerate(mazes):
            prefix = f"order{args.order}_maze{i}"
            if maze_data['is_perfect']:
                suffix = "perfect"
                perfect_count += 1
            else:
                suffix = "imperfect"

            # 控制台打印
            if args.print:
                print(f"\n迷宫 #{i + 1} (阶数 {args.order}, 尺寸 {maze_data['size']}x{maze_data['size']})")
                print(f"状态: {'完美迷宫' if maze_data['is_perfect'] else '非完美迷宫'}")
                MazeGenerator.print_maze(maze_data['maze'])

            # 保存为文本文件
            if args.format in ['text', 'all']:
                text_path = os.path.join(args.output_dir, f"{prefix}_{suffix}.txt")
                MazeGenerator.save_as_text(maze_data['maze'], text_path)

            # 保存为图像
            if args.format in ['image', 'all']:
                img_path = os.path.join(args.output_dir, f"{prefix}_{suffix}.png")
                MazeGenerator.save_as_image(maze_data['maze'], img_path)

            # 保存为JSON
            if args.format in ['json', 'all']:
                json_path = os.path.join(args.output_dir, f"{prefix}_{suffix}.json")
                MazeGenerator.save_as_json(maze_data, json_path)
            print(f"Saved: {i}")
    else:
        print(f"Discarded Data: {len(mazes)}")

    # 打印统计信息
    print(f"\nGenerate Completed!")
    print(f"Total               : {len(mazes)}")
    print(f"Perfect Maze Percent: {perfect_count} ({perfect_count / len(mazes) * 100:.1f}%)")
    if args.format != "no":
        print(f"Output to           : {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()