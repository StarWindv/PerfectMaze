import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PerfectMaze.HilbertMaze import HilbertMaze
from PerfectMaze import is_perfect_maze


logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] <%(levelname)s> | %(message)s',
    handlers=[
        logging.FileHandler("maze_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class Config:
    NOISE_DIM = 100
    EMBEDDING_DIM = 16
    HIDDEN_DIM = 384
    MAX_ORDER = 7

    BATCH_SIZE = 64
    EPOCHS = 1000
    LR_G = 0.0002
    LR_D = 0.0002
    BETAS = (0.5, 0.999)
    SAMPLE_INTERVAL = 25
    CHECKPOINT_INTERVAL = 100
    BATCHES_PER_EPOCH = 100

class MazeGenerator:
    

    def __init__(self, max_order=7):
        """初始化迷宫生成器

        Args:
            max_order (int): 最大迷宫阶数
        """
        self.max_order = max_order
        self.size_map = {order:(2 ** (order + 1) + 1) for order in range(max_order + 1)}

    @staticmethod
    def generate_batch(order, batch_size):
        """生成指定阶数的一个批次迷宫

        Args:
            order (int): 迷宫阶数
            batch_size (int): 批次大小

        Returns:
            tuple: (迷宫张量, 阶数张量)
        """
        
        mazes = []

        for _ in range(batch_size):
            creator = HilbertMaze(order=order)
            maze, _, _ = creator.generate()
            mazes.append(maze)

        
        mazes = np.array(mazes)
        mazes_tensor = torch.tensor(mazes, dtype=torch.float32)
        orders_tensor = torch.full((batch_size,), order, dtype=torch.long)

        return mazes_tensor, orders_tensor


class Generator(nn.Module):
    """生成器网络

    Attributes:
        embed - nn.Embedding : 阶数嵌入层
        main  - nn.Sequential: 主网络结构
    """

    def __init__(self, noise_dim=100, embed_dim=16, hidden_dim=256, max_order=7):
        """初始化生成器

        Args:
            noise_dim (int, optional): 噪声维度. Defaults to 100.
            embed_dim (int, optional): 嵌入维度. Defaults to 16.
            hidden_dim (int, optional): 隐藏层维度. Defaults to 256.
            max_order (int, optional): 最大阶数. Defaults to 7.
        """
        super(Generator, self).__init__()
        self.max_order = max_order
        self.embed = nn.Embedding(max_order + 1, embed_dim)

        
        self.fc_layers = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(inplace=True),
        )

        
        self.output_layers = nn.ModuleDict()
        for order in range(max_order + 1):
            size = (2 ** (order + 1) + 1) ** 2
            self.output_layers[str(order)] = nn.Sequential(
                nn.Linear(hidden_dim * 4, size),
                nn.Sigmoid()
            )

    def forward(self, noise, orders):
        """前向传播

        Args:
            noise (Tensor): 噪声向量
            orders (Tensor): 阶数标签

        Returns:
            Tensor: 生成的迷宫矩阵
        """
        
        embed = self.embed(orders)

        
        x = torch.cat([noise, embed], dim=1)

        
        x = self.fc_layers(x)

        
        outputs = []
        for i, order in enumerate(orders):
            order_key = str(order.item())
            output_layer = self.output_layers[order_key]
            output = output_layer(x[i])

            
            size = (2 ** (order.item() + 1) + 1)
            output = output.view(size, size)
            outputs.append(output)

        
        return torch.stack(outputs)


class Discriminator(nn.Module):
    """判别器网络

    Attributes:
        embed - nn.Embedding : 阶数嵌入层
        main  - nn.Sequential: 主网络结构
    """

    def __init__(self, embed_dim=16, hidden_dim=256, max_order=7):
        """初始化判别器

        Args:
            embed_dim (int, optional): 嵌入维度. Defaults to 16.
            hidden_dim (int, optional): 隐藏层维度. Defaults to 256.
            max_order (int, optional): 最大阶数. Defaults to 7.
        """
        super(Discriminator, self).__init__()
        self.max_order = max_order
        self.embed = nn.Embedding(max_order + 1, embed_dim)

        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        
        self.input_layers = nn.ModuleDict()
        for order in range(max_order + 1):
            size = (2 ** (order + 1) + 1) ** 2
            self.input_layers[str(order)] = nn.Sequential(
                nn.Linear(size + embed_dim, hidden_dim * 4),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, mazes, orders):
        """前向传播

        Args:
            mazes (Tensor): 迷宫矩阵
            orders (Tensor): 阶数标签

        Returns:
            Tensor: 真实性概率
        """
        
        outputs = []
        for i, maze in enumerate(mazes):
            order = orders[i].item()
            order_key = str(order)

            
            maze_flat = maze.flatten()

            
            embed = self.embed(orders[i].unsqueeze(0))

            
            x = torch.cat([maze_flat, embed.squeeze(0)])

            
            input_layer = self.input_layers[order_key]
            x = input_layer(x)

            
            x = self.fc_layers(x)
            outputs.append(x)

        
        return torch.stack(outputs)


class GANTrainer:
    """GAN训练器

    Attributes:
        generator (Generator): 生成器
        discriminator (Discriminator): 判别器
        optimizer_G (optim.Adam): 生成器优化器
        optimizer_D (optim.Adam): 判别器优化器
        criterion (nn.BCELoss): 损失函数
        maze_generator (MazeGenerator): 迷宫生成器
        current_epoch (int): 当前训练轮数
    """

    def __init__(self, train_config):
        """初始化训练器

        Args:
            train_config (Config): 配置对象
        """
        self.config = train_config

        # 初始化网络
        self.generator = Generator(
            noise_dim=train_config.NOISE_DIM,
            embed_dim=train_config.EMBEDDING_DIM,
            hidden_dim=train_config.HIDDEN_DIM,
            max_order=train_config.MAX_ORDER
        ).to(device)

        self.discriminator = Discriminator(
            embed_dim=train_config.EMBEDDING_DIM,
            hidden_dim=train_config.HIDDEN_DIM,
            max_order=train_config.MAX_ORDER
        ).to(device)

        # 初始化优化器
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=train_config.LR_G,
            betas=train_config.BETAS
        )

        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=train_config.LR_D,
            betas=train_config.BETAS
        )

        self.scheduler_G = optim.lr_scheduler.StepLR(
            self.optimizer_G,
            step_size=100,
            gamma=0.9
        )

        self.scheduler_D = optim.lr_scheduler.StepLR(
            self.optimizer_D,
            step_size=100,
            gamma=0.9
        )

        self.criterion = nn.BCELoss()

        # 迷宫生成器
        self.maze_generator = MazeGenerator(max_order=train_config.MAX_ORDER)

        # 训练状态
        self.current_epoch = 0
        self.best_g_loss = float('inf')
        self.best_d_loss = float('inf')

        # 加载检查点
        self.load_checkpoint()

    def train(self, epochs):
        """训练模型

        Args:
            epochs (int): 训练轮数
        """
        logger.info(f"Starting training for {epochs} epochs")

        start_epoch = self.current_epoch
        end_epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            start_time = time.time()

            g_losses = []
            d_losses = []

            # 每个epoch中的批次训练
            for batch_idx in range(self.config.BATCHES_PER_EPOCH):
                # 随机选择阶数
                order = np.random.randint(2, self.config.MAX_ORDER + 1)
                # 决定最小训练阶数与最大阶数

                # 生成真实迷宫
                real_mazes, orders_tensor = self.maze_generator.generate_batch(
                    order, self.config.BATCH_SIZE
                )
                real_mazes = real_mazes.to(device)
                orders_tensor = orders_tensor.to(device)

                # 真实和虚假标签
                real_labels = torch.ones(self.config.BATCH_SIZE, 1, device=device)
                fake_labels = torch.zeros(self.config.BATCH_SIZE, 1, device=device)

                # 训练判别器
                self.optimizer_D.zero_grad()

                # 真实样本的损失
                real_output = self.discriminator(real_mazes, orders_tensor)
                d_loss_real = self.criterion(real_output, real_labels)

                # 生成虚假样本
                noise = torch.randn(self.config.BATCH_SIZE, self.config.NOISE_DIM, device=device)
                fake_mazes = self.generator(noise, orders_tensor)

                # 虚假样本的损失
                fake_output = self.discriminator(fake_mazes.detach(), orders_tensor)
                d_loss_fake = self.criterion(fake_output, fake_labels)

                # 判别器总损失
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # 训练生成器
                self.optimizer_G.zero_grad()

                fake_output = self.discriminator(fake_mazes, orders_tensor)
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.optimizer_G.step()

                # 记录损失
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

                # 定期记录
                if batch_idx % 10 == 0:
                    avg_g_loss = np.mean(g_losses[-10:]) if len(g_losses) > 10 else g_loss.item()
                    avg_d_loss = np.mean(d_losses[-10:]) if len(d_losses) > 10 else d_loss.item()

                    logger.info(
                        f"[Epoch {epoch}/{end_epoch}] "
                        f"[Batch {batch_idx}/{self.config.BATCHES_PER_EPOCH}] "
                        f"Order: {order} | "
                        f"D_loss: {avg_d_loss:.4f} | "
                        f"G_loss: {avg_g_loss:.4f} | "
                        f"LR_G: {self.scheduler_G.get_last_lr()[0]:.6f} | "  # 新增：记录学习率
                        f"LR_D: {self.scheduler_D.get_last_lr()[0]:.6f}"  # 新增：记录学习率
                    )

            # 计算平均损失
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)

            # 更新学习率 - 新增部分
            self.scheduler_G.step()
            self.scheduler_D.step()

            # 保存最佳模型
            if avg_g_loss < self.best_g_loss:
                self.best_g_loss = avg_g_loss
                self.save_checkpoint(best=True)

            # 定期保存检查点
            if epoch % self.config.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint()

            # 定期生成样本
            if epoch % self.config.SAMPLE_INTERVAL == 0:
                self.sample_mazes(epoch)

            # 记录epoch信息
            epoch_time = time.time() - start_time
            logger.info(
                f"[Epoch {epoch}/{end_epoch}] completed | "
                f"Time: {epoch_time:.2f}s | "
                f"D_loss: {avg_d_loss:.4f} | "
                f"G_loss: {avg_g_loss:.4f} | "
                f"LR_G: {self.scheduler_G.get_last_lr()[0]:.6f} | "  # 新增：记录学习率
                f"LR_D: {self.scheduler_D.get_last_lr()[0]:.6f}"  # 新增：记录学习率
            )

    def sample_mazes(self, epoch, num_samples=5):
        """生成迷宫样本并保存

        Args:
            epoch (int): 当前轮数
            num_samples (int, optional): 每个阶数的样本数量. Defaults to 5.
        """
        self.generator.eval()

        with torch.no_grad():
            for order in range(self.config.MAX_ORDER + 1):
                
                noise = torch.randn(num_samples, self.config.NOISE_DIM, device=device)
                orders = torch.full((num_samples,), order, dtype=torch.long, device=device)

                
                generated = self.generator(noise, orders)

                
                generated_np = generated.cpu().numpy()

                
                for i in range(num_samples):
                    maze = generated_np[i]
                    
                    binary_maze = (maze > 0.5).astype(np.int32)
                    
                    is_perfect = is_perfect_maze(binary_maze)
                    
                    filename = f"samples/epoch{epoch}_order{order}_maze{i}_{'perfect' if is_perfect else 'imperfect'}.txt"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)

                    np.savetxt(filename, binary_maze, fmt="%d")

        self.generator.train()
        logger.info(f"Saved generated samples for epoch {epoch}\n")

    def save_checkpoint(self, best=False):
        """保存训练检查点

        Args:
            best (bool, optional): 是否保存为最佳模型. Defaults to False.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_G_state': self.optimizer_G.state_dict(),
            'optimizer_D_state': self.optimizer_D.state_dict(),
            'scheduler_G_state': self.scheduler_G.state_dict(),  # 新增：保存调度器状态
            'scheduler_D_state': self.scheduler_D.state_dict(),  # 新增：保存调度器状态
            'best_g_loss': self.best_g_loss,
        }

        if best:
            filename = f"checkpoints/best_model.pth"
        else:
            filename = f"checkpoints/checkpoint_epoch_{self.current_epoch}.pth"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(checkpoint, filename)
        logger.info(f"Saved {'best ' if best else ''}checkpoint at epoch {self.current_epoch}")

    def load_checkpoint(self, checkpoint_path=None):
        """加载训练检查点

        Args:
            checkpoint_path (str, optional): 检查点路径.
                Defaults to None (加载最新的检查点).
        """
        if checkpoint_path is None:
            # 尝试加载最佳模型
            if os.path.exists("checkpoints"):
                best_path = "checkpoints/best_model.pth"
                if os.path.exists(best_path):
                    checkpoint = torch.load(best_path, map_location=device, weights_only=False)

                    self.generator.load_state_dict(checkpoint['generator_state'])
                    self.discriminator.load_state_dict(checkpoint['discriminator_state'])
                    self.current_epoch = checkpoint['epoch']
                    self.best_g_loss = checkpoint['best_g_loss']

                    logger.info(f"Loaded best model from {best_path} (epoch {self.current_epoch})")
                    return True

                checkpoints = [f for f in os.listdir("checkpoints") if f.startswith("checkpoint")]
                if checkpoints:
                    # 找到最新的检查点
                    checkpoints.sort(key=lambda x: int(x.split('checkpoint_epoch_')[1].split('.')[0]))
                    checkpoint_path = os.path.join("checkpoints", checkpoints[-1])

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        else:
            logger.info("No checkpoint found. Starting from scratch.")
            return False

        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state'])

        # 新增：加载调度器状态
        if 'scheduler_G_state' in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state'])
        if 'scheduler_D_state' in checkpoint:
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state'])

        self.current_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
        self.best_g_loss = checkpoint['best_g_loss']

        logger.info(f"Loaded checkpoint from {checkpoint_path} (resuming from epoch {self.current_epoch})")
        return True

    def generate_maze(self, order, num_samples=1, threshold=0.5):
        """生成指定阶数的迷宫

        Args:
            order (int): 迷宫阶数 (0-7)
            num_samples (int, optional): 生成数量. Defaults to 1.
            threshold (float, optional): 二值化阈值. Defaults to 0.5.

        Returns:
            list: 生成的迷宫列表 (np.array)
        """
        if not (0 <= order <= self.config.MAX_ORDER):
            raise ValueError(f"Order must be between 0 and {self.config.MAX_ORDER}")

        self.generator.eval()

        with torch.no_grad():
            
            noise = torch.randn(num_samples, self.config.NOISE_DIM, device=device)
            orders = torch.full((num_samples,), order, dtype=torch.long, device=device)

            
            generated = self.generator(noise, orders)

            generated_np = generated.cpu().numpy()
            
            results = []
            for i in range(num_samples):
                maze = generated_np[i]
                binary_maze = (maze > threshold).astype(np.int32)
                results.append(binary_maze)

        self.generator.train()
        return results


def main():
    config = Config()
    trainer = GANTrainer(config)
    try:
        trainer.train(config.EPOCHS)
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving checkpoint...")
        trainer.save_checkpoint()
        logger.info("Checkpoint saved. Exiting.")


if __name__ == "__main__":
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    main()