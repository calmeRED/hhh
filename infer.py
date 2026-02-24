# -*- coding: utf-8 -*-
# infer.py
import os
import pdb
import re
import argparse
import numpy as np
import torch

from trainer import Trainer
from utils.utils_config import get_config
from config.base_config import config


def find_latest_checkpoint(run_dir):
    """
    在 run_dir 下找到 episode 数最大的 checkpoint_xxx.pth
    """
    ckpts = []
    for f in os.listdir(run_dir):
        if f.startswith("checkpoint_") and f.endswith(".pth"):
            m = re.search(r"checkpoint_(\d+)\.pth", f)
            if m:
                ckpts.append((int(m.group(1)), os.path.join(run_dir, f)))

    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoint_*.pth found in {run_dir}")

    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1], ckpts[-1][0]


def load_latest_buffer_if_exists(trainer, run_dir, episode_id):
    """
    （可选）加载 replay buffer
    """
    buf_path = os.path.join(run_dir, f"replay_buffer_{episode_id}.pkl")
    if os.path.exists(buf_path):
        print(f"[Infer] Loading replay buffer: {buf_path}")
        trainer.load_buffer(buf_path)
    else:
        print("[Infer] No replay buffer found, skip.")


@torch.no_grad()
def run_inference(trainer, num_episodes=1, render=False):
    """
    运行推理 episode
    """
    print(f"[Infer] Start inference for {num_episodes} episode(s)")
    all_rewards = []

    for ep in range(num_episodes):
        rewards = trainer.evaluate_episode()
        all_rewards.append(rewards)
        print(f"[Infer] Episode {ep:03d} | rewards = {rewards}, avg = {np.mean(rewards):.4f}")

    print("=" * 60)
    print("[Infer] Done.")
    print(f"[Infer] Mean reward over episodes: {np.mean(all_rewards):.4f}")
    return all_rewards


def main(run_dir, episodes, load_buffer):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        default=run_dir,
        type=str,
        help="训练输出目录，例如 runs/20260130-1138"
    )
    parser.add_argument(
        "--episodes",
        default=episodes,
        type=int,
        help="推理 episode 数"
    )
    parser.add_argument(
        "--load_buffer",
        default=load_buffer,
        action="store_true",
        help="是否加载 replay buffer（通常不需要）"
    )

    args = parser.parse_args()

    # ===== 1. config =====
    my_config = get_config(config)

    # 推理时强制关闭探索（双保险）
    my_config["epsilon"] = 0.0
    my_config["explore"] = False

    # ===== 2. 创建 Trainer =====
    trainer = Trainer(my_config)

    # ===== 3. 加载 checkpoint =====
    ckpt_path, episode_id = find_latest_checkpoint(args.run_dir)
    print(f"[Infer] Loading checkpoint: {ckpt_path}")
    trainer.load_checkpoint(ckpt_path)

    # ===== 4. （可选）加载 buffer =====
    if args.load_buffer:
        load_latest_buffer_if_exists(trainer, args.run_dir, episode_id)
    pdb.set_trace()
    # ===== 5. 推理 =====
    run_inference(trainer, num_episodes=args.episodes)


if __name__ == "__main__":
    main("runs/20260130-1138", 5, True)