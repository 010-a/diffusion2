import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# use like : python plot_loss.py "output/2025-10-22_17-25-28/logs/training_log.csv"
def plot_loss_curve(log_file_path: str):
    """
    从CSV日志文件中读取数据，绘制训练损失曲线，并标记出最佳epoch。

    Args:
        log_file_path (str): CSV日志文件的路径。
    """
    # --- 1. 数据加载与验证 ---
    if not os.path.exists(log_file_path):
        print(f"错误：指定的日志文件不存在。\n路径: '{log_file_path}'")
        return

    try:
        log_data = pd.read_csv(log_file_path)
    except Exception as e:
        print(f"错误：使用pandas读取CSV文件时出错: {e}")
        return

    required_columns = ['epoch', 'avg_loss']
    if not all(col in log_data.columns for col in required_columns):
        print(f"错误：日志文件必须包含以下列: {required_columns}")
        return

    if log_data.empty:
        print("错误：日志文件为空，无法进行绘图。")
        return

    # --- 2. 分析数据以定位最佳点 ---
    # 使用 idxmin() 方法找到第一次出现最小损失值的行索引
    best_idx = log_data['avg_loss'].idxmin()
    best_epoch_data = log_data.loc[best_idx]

    best_epoch = int(best_epoch_data['epoch'])
    min_loss = best_epoch_data['avg_loss']

    print("\n" + "=" * 50)
    print("训练日志分析摘要")
    print("=" * 50)
    print(f"日志文件: {os.path.basename(log_file_path)}")
    print(f"总计 Epochs: {len(log_data)}")
    print(f"最低平均损失: {min_loss:.6f}")
    print(f"对应的 Epoch: {best_epoch}")
    print("=" * 50 + "\n")

    # --- 3. 绘图与可视化 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    # 绘制主曲线
    ax.plot(log_data['epoch'], log_data['avg_loss'], marker='.', linestyle='-', markersize=5,
            label='Average Training Loss', color='royalblue')

    # 标记最佳点
    ax.plot(best_epoch, min_loss, 'o', markersize=8, color='red', label=f'Best Epoch ({best_epoch})', zorder=5)

    # 添加注释
    ax.annotate(
        f'Min Loss: {min_loss:.4f}',
        xy=(best_epoch, min_loss),
        xytext=(0, 20),  # 文本偏移量
        textcoords='offset points',
        ha='center',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
    )

    # --- 4. 设置图表属性并保存 ---
    ax.set_title('Training Loss Curve Analysis', fontsize=16, weight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.tight_layout()

    # 将图表保存在与日志文件相同的目录下
    save_path = os.path.splitext(log_file_path)[0] + '.png'
    try:
        plt.savefig(save_path, dpi=300)
        print(f"✔ 损失曲线图已成功保存至:\n'{save_path}'\n")
    except Exception as e:
        print(f"错误：保存图像时出错: {e}")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从CSV日志文件绘制训练损失曲线图并找出最佳模型位置。")
    parser.add_argument('log_file', type=str,
                        help='包含训练日志的 .csv 文件路径。')
    args = parser.parse_args()

    plot_loss_curve(args.log_file)