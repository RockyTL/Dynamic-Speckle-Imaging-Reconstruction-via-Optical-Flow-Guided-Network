import copy
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from collections import defaultdict
from utils.Visual_utils import *


def process_batch_losses(object_losses, epoch_losses):
    """
    处理单个批次的损失。

    参数:
        object_losses (list): 包含每帧损失值的字典列表
        epoch_losses (defaultdict): 累积整个周期损失的字典

    返回:
        defaultdict: 批次损失字典
    """
    batch_losses = defaultdict(float)
    for obj_loss in object_losses:
        for key, value in obj_loss.items():
            epoch_losses[key] += value
            batch_losses[key] += value

    return batch_losses


def update_epoch_losses(epoch_losses, train_loader_len):
    """
    计算归一化的周期损失并创建用于报告的副本。

    参数:
        epoch_losses (defaultdict): 原始累积的周期损失
        train_loader_len (int): 训练加载器的长度
        num_frames (int): 处理的帧数

    返回:
        tuple: (归一化的周期损失, 用于绘图的副本, 原始周期损失的副本)
    """

    # 归一化周期损失
    for key in epoch_losses.keys():
        epoch_losses[key] /= train_loader_len

    return epoch_losses


def save_best_model(model, epoch_losses, best_loss, save_dir, epoch):
    """
    如果当前损失优于最佳损失，则保存模型。

    参数:
        model: 要保存的模型
        epoch_losses (dict): 当前周期的损失
        best_loss (float): 迄今为止的最佳损失
        save_dir (str): 保存模型的目录

    返回:
        float: 更新后的最佳损失
    """
    # 创建保存目录
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if epoch_losses['total_loss'] < best_loss:
        best_loss = epoch_losses['total_loss']
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'other_best_model.pth'))

    if (epoch + 1) % 5 == 0:
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), ckpt_path)

    return best_loss


def log_epoch_results(epoch, num_epochs, epoch_losses, best_loss, save_dir):
    """
    将周期结果记录到控制台和文件。

    参数:
        epoch (int): 当前周期
        num_epochs (int): 总周期数
        epoch_losses (dict): 归一化的周期损失
        epoch_losses_unnormalized (dict): 未归一化的周期损失
        best_loss (float): 迄今为止的最佳损失
        save_dir (str): 保存日志的目录
    """
    # 生成损失摘要
    epoch_losses_summary = epoch_loss_summary(epoch_losses)

    # 格式化并打印损失表格
    loss_table = format_epoch_loss_table(epoch, num_epochs, epoch_losses_summary, best_loss)
    print(loss_table)

    # 记录到文件
    log_file = os.path.join(save_dir, 'result_log.csv')
    if epoch == 0:
        with open(log_file, 'w', encoding='utf-8') as log:
            log.write("")

    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(loss_table)


def process_test_frame_losses(test_losses, loss_dict):
    """
    处理单个测试帧的损失。

    参数:
        test_losses (defaultdict): 累积的测试损失
        loss_dict (dict): 当前帧的损失字典
    """
    for key, val in loss_dict.items():
        # 只统计标量 loss
        if isinstance(val, (float, int)):
            test_losses[key] += val
        # 其余类型（list / dict）全部跳过
    return test_losses


def update_frame_losses(frame_losses, total_loss, loss_dict):
    """
    更新帧损失和总损失字典。

    参数:
        frame_losses (defaultdict): 每帧损失的列表字典
        total_loss (defaultdict): 累积损失的字典
        loss_dict (dict): 当前帧的损失字典
    """
    for key in loss_dict.keys():
        frame_losses[key].append(loss_dict[key])
        total_loss[key] += loss_dict[key]

    return frame_losses, total_loss


def process_test_results(test_losses, test_loader_len):
    """
    通过归一化损失并创建用于报告的副本来处理测试结果。

    参数:
        test_losses (defaultdict): 累积的测试损失
        test_loader_len (int): 测试加载器的长度
        num_frames (int): 处理的帧数

    返回:
        tuple: (归一化的测试损失, 未归一化的测试损失)
    """
    # 归一化测试损失
    for key in test_losses.keys():
        test_losses[key] /= test_loader_len

    return test_losses


def log_test_results(test_losses, save_dir):
    """
    将测试结果记录到控制台和文件。

    参数:
        test_losses (dict): 归一化的测试损失
        test_losses_unnormalized (dict): 未归一化的测试损失
        save_dir (str): 保存日志的目录
    """
    # 格式化最终损失表格
    final_loss_table = format_final_loss_table(test_losses)

    # 打印到控制台
    print(final_loss_table)

    # 记录到文件
    log_file = os.path.join(save_dir, 'result_log.csv')
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write("\n" + final_loss_table)