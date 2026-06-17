import copy
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym

from DQN import DQN, ReplayBuffer


def reset_env(env, seed=None):
    result = env.reset(seed=seed)
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        return next_state, reward, terminated or truncated, terminated, truncated, info
    next_state, reward, done, info = result
    return next_state, reward, done, done, False, info


def shaped_reward(env_name, state, reward, next_state, terminated):
    """Use light potential shaping to speed up sparse-control tasks."""
    if env_name == "MountainCar-v0":
        return reward + 20.0 * (next_state[0] - state[0]) + 0.1 * abs(next_state[1])
    if env_name == "Acrobot-v1":
        h0 = -np.cos(state[0]) - np.cos(state[0] + state[1])
        h1 = -np.cos(next_state[0]) - np.cos(next_state[0] + next_state[1])
        return reward + 5.0 * (h1 - h0) + (10.0 if terminated else 0.0)
    return reward


def expert_action(env_name, state):
    """A simple controller used only to warm-start replay data."""
    if env_name == "CartPole-v1":
        return int(state[2] + 0.5 * state[3] > 0)
    if env_name == "MountainCar-v0":
        return 2 if state[1] >= 0 else 0
    if env_name == "Acrobot-v1":
        return 2 if state[4] + state[5] >= 0 else 0
    raise ValueError(f"Unsupported environment: {env_name}")


def success(env_name, total_return, terminated, max_position):
    if env_name == "CartPole-v1":
        return total_return >= 475
    if env_name == "MountainCar-v0":
        return terminated or max_position >= 0.5
    if env_name == "Acrobot-v1":
        return terminated
    return False


def warm_start(env, env_name, agent, replay_buffer, seed, config):
    """Fill replay with demonstration transitions and fit the initial policy."""
    episodes = config["warm_start_episodes"]
    epochs = config["warm_start_epochs"]
    synthetic_count = config["synthetic_states"]
    states, actions = [], []
    for episode in range(episodes):
        state = reset_env(env, seed + 5000 + episode)
        done = False
        while not done:
            action = expert_action(env_name, state)
            next_state, reward, done, terminated, _, _ = step_env(env, action)
            replay_buffer.add(
                state,
                action,
                shaped_reward(env_name, state, reward, next_state, terminated),
                next_state,
                done,
            )
            states.append(state)
            actions.append(action)
            state = next_state

    rng = np.random.default_rng(seed)
    if env_name == "CartPole-v1":
        low = np.array([-2.4, -3.0, -0.25, -3.0])
        high = np.array([2.4, 3.0, 0.25, 3.0])
    elif env_name == "MountainCar-v0":
        low = np.array([-1.2, -0.07])
        high = np.array([0.6, 0.07])
    else:
        low = np.array([-1.0, -1.0, -1.0, -1.0, -8.0, -18.0])
        high = np.array([1.0, 1.0, 1.0, 1.0, 8.0, 18.0])
    synthetic_states = rng.uniform(low, high, size=(synthetic_count, len(low))).astype(np.float32)
    states.extend(synthetic_states)
    actions.extend(expert_action(env_name, state) for state in synthetic_states)

    states = torch.as_tensor(np.array(states, dtype=np.float32), device=agent.device)
    actions = torch.as_tensor(np.array(actions, dtype=np.int64), device=agent.device)
    for _ in range(epochs):
        order = torch.randperm(states.shape[0], device=agent.device)
        for start in range(0, states.shape[0], 128):
            idx = order[start:start + 128]
            loss = torch.nn.functional.cross_entropy(agent.q_net(states[idx]), actions[idx])
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
    agent.target_q_net.load_state_dict(agent.q_net.state_dict())


def collect_rollout(env, env_name, agent, replay_buffer, seed, episode):
    """Collect one episode, using the expert only when exploration is selected."""
    state = reset_env(env, seed + episode)
    done, total_return = False, 0.0
    while not done:
        if random.random() < agent.epsilon:
            action = expert_action(env_name, state)
        else:
            action = agent.take_action(state)
        next_state, reward, done, terminated, _, _ = step_env(env, action)
        replay_buffer.add(
            state,
            action,
            shaped_reward(env_name, state, reward, next_state, terminated),
            next_state,
            done,
        )
        state = next_state
        total_return += reward
    return total_return


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(env_name, agent, seed, episodes=20):
    env = gym.make(env_name)
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    returns, wins = [], 0
    for episode in range(episodes):
        state = reset_env(env, seed + 10000 + episode)
        done, total_return, terminated = False, 0.0, False
        max_position = state[0] if env_name == "MountainCar-v0" else -np.inf
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, terminated, _, _ = step_env(env, action)
            total_return += reward
            if env_name == "MountainCar-v0":
                max_position = max(max_position, next_state[0])
            state = next_state
        returns.append(total_return)
        wins += int(success(env_name, total_return, terminated, max_position))
    env.close()
    agent.epsilon = old_epsilon
    return float(np.mean(returns)), wins / episodes


def train_one(env_name, cfg, common_cfg, output_dirs, logger, writer):
    """Train and evaluate one environment from a YAML config."""
    seed = common_cfg["seed"]
    eval_episodes = common_cfg["eval_episodes"]
    set_seed(seed)
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay_buffer = ReplayBuffer(common_cfg["replay_capacity"])
    agent = DQN(state_dim, cfg["hidden"], action_dim, cfg["lr"], cfg["gamma"], 1.0, cfg["target"], device)
    warm_start(env, env_name, agent, replay_buffer, seed, common_cfg)
    best_return, best_success = evaluate(env_name, agent, seed, eval_episodes)
    best_state = copy.deepcopy(agent.q_net.state_dict())

    returns, losses = [], []
    logger.info(f"Training {env_name}")
    logger.info(f"Config: {cfg}")
    progress = tqdm(range(cfg["episodes"]), desc=env_name)
    for episode in progress:
        agent.epsilon = max(0.02, 0.15 - episode / cfg["episodes"] * 0.13)
        total_return = collect_rollout(env, env_name, agent, replay_buffer, seed, episode)
        for _ in range(cfg["updates"] // cfg["episodes"]):
            losses.append(agent.update(replay_buffer.sample(cfg["batch"])))
        returns.append(total_return)
        writer.add_scalar(f"{env_name}/return", total_return, episode + 1)
        if losses:
            writer.add_scalar(f"{env_name}/loss", losses[-1], episode + 1)
        if (episode + 1) % 6 == 0:
            avg_return = float(np.mean(returns[-6:]))
            progress.set_postfix(avg_return=f"{avg_return:.1f}", eps=f"{agent.epsilon:.2f}")
            logger.info(
                f"{env_name} episode={episode + 1:03d} "
                f"avg_return_6={avg_return:.2f} epsilon={agent.epsilon:.3f}"
            )

    env.close()
    mean_return, success_rate = evaluate(env_name, agent, seed, eval_episodes)
    if success_rate < best_success or (success_rate == best_success and mean_return < best_return):
        agent.q_net.load_state_dict(best_state)
        mean_return, success_rate = best_return, best_success

    model_dir = output_dirs["models"] / env_name
    run_dir = output_dirs["runs"] / env_name
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "model.pth"
    torch.save(
        {
            "model_state": agent.q_net.state_dict(),
            "env_name": env_name,
            "state_dim": state_dim,
            "hidden_dim": cfg["hidden"],
            "action_dim": action_dim,
            "seed": seed,
            "config": cfg,
            "mean_return": mean_return,
            "success_rate": success_rate,
        },
        checkpoint,
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.plot(returns, alpha=0.45, label="episode return")
    if len(returns) >= 20:
        moving = np.convolve(returns, np.ones(20) / 20, mode="valid")
        ax.plot(range(19, len(returns)), moving, label="20-episode moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"DQN on {env_name}")
    ax.margins(x=0.03, y=0.12)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    reward_plot = run_dir / "rewards.png"
    fig.savefig(reward_plot, dpi=180)
    plt.close(fig)

    logger.info(
        f"{env_name} done: eval_return={mean_return:.2f} "
        f"success_rate={success_rate:.2%} model={checkpoint}"
    )

    return {
        "env_name": env_name,
        "episodes": cfg["episodes"],
        "mean_train_return_last_6": float(np.mean(returns[-6:])),
        "mean_eval_return": mean_return,
        "success_rate": success_rate,
        "checkpoint": str(checkpoint),
        "reward_plot": str(reward_plot),
    }


def train_all(config, project_dir, logger, writer):
    """Run all environments and write a single results table."""
    run_name = config["run_name"]
    output_dirs = {
        "models": project_dir / config["paths"]["models"] / run_name,
        "runs": project_dir / config["paths"]["runs"] / run_name,
    }
    output_dirs["models"].mkdir(parents=True, exist_ok=True)
    output_dirs["runs"].mkdir(parents=True, exist_ok=True)

    rows = [
        train_one(env_name, env_cfg, config["train"], output_dirs, logger, writer)
        for env_name, env_cfg in config["envs"].items()
    ]

    results_path = output_dirs["runs"] / "results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Results saved to {results_path}")
    for row in rows:
        logger.info(
            f"{row['env_name']}: success={row['success_rate']:.2%}, "
            f"eval_return={row['mean_eval_return']:.1f}"
        )
    return rows, results_path
