#!/usr/bin/env python3
"""
解析训练日志并绘制训练曲线

用法:
    python plot_training_log.py [日志文件路径] [输出图片路径] [起始步数]
    
示例:
    python plot_training_log.py nohup_log.txt training_curves.png 1000
    python plot_training_log.py nohup_log.txt training_curves.png 0  # 从第0步开始
    python plot_training_log.py  # 使用默认文件名，从第1000步开始
"""
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_file):
    """解析日志文件，提取step、loss、validation loss和HellaSwag accuracy"""
    
    steps = []
    losses = []
    
    val_steps = []
    val_losses = []
    
    hellaswag_steps = []
    hellaswag_accs = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        current_step = 0
        for line in f:
            # 匹配训练步骤行: step     0 | loss: 10.955029 | ...
            step_match = re.match(r'step\s+(\d+)\s+\|\s+loss:\s+([\d.]+)', line)
            if step_match:
                step_num = int(step_match.group(1))
                loss_val = float(step_match.group(2))
                steps.append(step_num)
                losses.append(loss_val)
                current_step = step_num
                continue
            
            # 匹配验证loss行: validation loss: 4.1614
            val_match = re.match(r'validation loss:\s+([\d.]+)', line)
            if val_match:
                val_loss = float(val_match.group(1))
                val_steps.append(current_step)
                val_losses.append(val_loss)
                continue
            
            # 匹配HellaSwag行: HellaSwag accuracy: 2589/10042=0.2578
            hellaswag_match = re.match(r'HellaSwag accuracy:\s+\d+/\d+=([\d.]+)', line)
            if hellaswag_match:
                acc = float(hellaswag_match.group(1))
                hellaswag_steps.append(current_step)
                hellaswag_accs.append(acc)
                continue
    
    return {
        'train_steps': steps,
        'train_losses': losses,
        'val_steps': val_steps,
        'val_losses': val_losses,
        'hellaswag_steps': hellaswag_steps,
        'hellaswag_accs': hellaswag_accs
    }

def plot_training_curves(data, output_file='training_curves.png', start_step=1000):
    """绘制训练曲线
    
    Args:
        data: 解析后的数据字典
        output_file: 输出文件名
        start_step: 起始步数，只绘制 >= start_step 的数据
    """
    
    # 过滤数据，只保留 >= start_step 的数据
    filtered_data = {
        'train_steps': [],
        'train_losses': [],
        'val_steps': [],
        'val_losses': [],
        'hellaswag_steps': [],
        'hellaswag_accs': []
    }
    
    # 过滤训练数据
    for step, loss in zip(data['train_steps'], data['train_losses']):
        if step >= start_step:
            filtered_data['train_steps'].append(step)
            filtered_data['train_losses'].append(loss)
    
    # 过滤验证数据
    for step, loss in zip(data['val_steps'], data['val_losses']):
        if step >= start_step:
            filtered_data['val_steps'].append(step)
            filtered_data['val_losses'].append(loss)
    
    # 过滤HellaSwag数据
    for step, acc in zip(data['hellaswag_steps'], data['hellaswag_accs']):
        if step >= start_step:
            filtered_data['hellaswag_steps'].append(step)
            filtered_data['hellaswag_accs'].append(acc)
    
    # 使用过滤后的数据
    data = filtered_data
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 绘制训练loss
    ax1 = axes[0]
    ax1.plot(data['train_steps'], data['train_losses'], 'b-', alpha=0.6, linewidth=0.8, label='Training Loss')
    if data['val_steps']:
        ax1.scatter(data['val_steps'], data['val_losses'], c='r', s=50, zorder=5, label='Validation Loss')
        ax1.plot(data['val_steps'], data['val_losses'], 'r--', alpha=0.5, linewidth=1.5)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 绘制验证loss（单独）
    ax2 = axes[1]
    if data['val_steps']:
        ax2.plot(data['val_steps'], data['val_losses'], 'r-o', linewidth=2, markersize=6, label='Validation Loss')
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Validation Loss', fontsize=12)
        ax2.set_title('Validation Loss Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
    
    # 绘制HellaSwag准确率
    ax3 = axes[2]
    if data['hellaswag_steps']:
        ax3.plot(data['hellaswag_steps'], data['hellaswag_accs'], 'g-o', linewidth=2, markersize=6, label='HellaSwag Accuracy')
        ax3.set_xlabel('Step', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('HellaSwag Accuracy Curve', fontsize=14, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        # 设置y轴范围为0-1
        ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存到: {output_file}')
    
    # 打印统计信息
    print(f'\n统计信息 (从step {start_step}开始):')
    print(f'训练步数: {len(data["train_steps"])}')
    if data['train_steps']:
        print(f'步数范围: {data["train_steps"][0]} - {data["train_steps"][-1]}')
    if data['train_losses']:
        print(f'训练Loss - 最小: {min(data["train_losses"]):.4f}, 最大: {max(data["train_losses"]):.4f}, 最终: {data["train_losses"][-1]:.4f}')
    print(f'验证次数: {len(data["val_steps"])}')
    if data['val_losses']:
        print(f'验证Loss - 最小: {min(data["val_losses"]):.4f}, 最大: {max(data["val_losses"]):.4f}, 最终: {data["val_losses"][-1]:.4f}')
    print(f'HellaSwag评估次数: {len(data["hellaswag_steps"])}')
    if data['hellaswag_accs']:
        print(f'HellaSwag准确率 - 最小: {min(data["hellaswag_accs"]):.4f}, 最大: {max(data["hellaswag_accs"]):.4f}, 最终: {data["hellaswag_accs"][-1]:.4f}')

if __name__ == '__main__':
    import sys
    
    # 默认参数
    log_file = 'nohup_log.txt'
    output_file = 'training_curves.png'
    start_step = 1000  # 默认从第1000步开始
    
    # 如果提供了命令行参数
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        start_step = int(sys.argv[3])
    
    print(f'解析日志文件: {log_file}')
    print(f'起始步数: {start_step}')
    data = parse_log(log_file)
    
    print(f'绘制训练曲线...')
    plot_training_curves(data, output_file, start_step=start_step)
    
    print('完成!')

