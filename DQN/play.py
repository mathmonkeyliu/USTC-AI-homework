import argparse
import random
import numpy as np
import torch
from DQN import Qnet
from tqdm import tqdm

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym


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


def is_success(env_name, total_return, terminated, max_position):
    if env_name == "CartPole-v1":
        return total_return >= 475
    if env_name == "MountainCar-v0":
        return terminated or max_position >= 0.5
    if env_name == "Acrobot-v1":
        return terminated
    return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint.")
    parser.add_argument("--model", required=True, help="Path to model.pth")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    env_name = checkpoint["env_name"]
    seed = checkpoint.get("seed", 42)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    render_mode = "human" if args.render else None
    env = gym.make(env_name, render_mode=render_mode)
    q_net = Qnet(checkpoint["state_dim"], checkpoint["hidden_dim"], checkpoint["action_dim"]).to(device)
    q_net.load_state_dict(checkpoint["model_state"])
    q_net.eval()

    def policy(state):
        """Greedy action for evaluation."""
        state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return q_net(state).argmax(dim=1).item()

    success_num, returns = 0, []
    for episode in tqdm(range(args.episodes), desc=env_name):
        state = reset_env(env, seed + 20000 + episode)
        done, terminated, total_return = False, False, 0.0
        max_position = state[0] if env_name == "MountainCar-v0" else -np.inf
        while not done:
            action = policy(state)
            state, reward, done, terminated, _, _ = step_env(env, action)
            total_return += reward
            if env_name == "MountainCar-v0":
                max_position = max(max_position, state[0])
        returns.append(total_return)
        success_num += int(is_success(env_name, total_return, terminated, max_position))

    env.close()
    print(f"Mean return: {np.mean(returns):.2f}")
    print(f"Success rate: {success_num / args.episodes:.2%}")


if __name__ == "__main__":
    main()


