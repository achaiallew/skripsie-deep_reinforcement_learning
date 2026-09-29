import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from minigrid.wrappers import ImgObsWrapper

# =====================================================
# Settings
# =====================================================

MODEL_PATH = "dqn_trained.pth"      # <-- change if necessary
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 256
EVAL_EPISODES = 10

# =====================================================
# Network
# =====================================================

class DQN(nn.Module):

    def __init__(self, inputSize, numActions, hiddenLayerSize=(128,128)):
        super().__init__()

        self.fc1 = nn.Linear(inputSize, hiddenLayerSize[0])
        self.fc2 = nn.Linear(hiddenLayerSize[0], hiddenLayerSize[1])
        self.fc3 = nn.Linear(hiddenLayerSize[1], numActions)

    def forward(self, x):
        x = x.to(DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# =====================================================
# Preprocessing
# =====================================================

def extractObjectInformation2(observation):
    # Try this first
    return observation[:,:,0]

    # If you want to compare with your old preprocessing,
    # comment the above line and uncomment below.

    # rows, cols, x = observation.shape
    # tmp = np.reshape(observation,[rows*cols*x,1],'F')[0:rows*cols]
    # return np.reshape(tmp,[rows,cols],'C')

def normalize(observation, max_value=10.0):
    return observation / max_value

def flatten(observation):
    return torch.from_numpy(
        observation.flatten()
    ).float().unsqueeze(0)

def preprocess(observation):
    return flatten(
        normalize(
            extractObjectInformation2(observation)
        )
    )

# =====================================================
# Action Selection
# =====================================================

def select_action(state, policy_net):

    with torch.no_grad():

        q_values = policy_net(state)

        print()
        print("State")
        print(state)

        print()
        print("Q-values")
        print(q_values)

        action = q_values.max(1).indices.view(1,1)

        print("Chosen:", action.item())

        return action

# =====================================================
# Evaluation
# =====================================================

def evaluate(policy_net):

    env = gym.make(
        "MiniGrid-Empty-8x8-v0",
        render_mode=None
    ).unwrapped

    env = ImgObsWrapper(env)

    successes = 0

    for ep in range(EVAL_EPISODES):

        obs, _ = env.reset()

        state = preprocess(obs)

        print("="*60)
        print("Episode", ep)
        print("="*60)

        for step in range(MAX_STEPS):

            action = select_action(state, policy_net).item()

            next_obs, reward, done, truncated, info = env.step(action)

            next_state = preprocess(next_obs)

            print("-----------------------")
            print("Step:", step)
            print("Action:", action)

            print("Step Count:", env.unwrapped.step_count)

            print("Agent Pos:", env.unwrapped.agent_pos)
            print("Agent Dir:", env.unwrapped.agent_dir)

            print("Reward:", reward)

            print("Raw obs equal:",
                  np.array_equal(obs,next_obs))

            print("Object layer equal:",
                  np.array_equal(
                      obs[:,:,0],
                      next_obs[:,:,0]
                  ))

            print("Processed state equal:",
                  torch.equal(state,next_state))

            print()

            if done:

                print("*** GOAL REACHED ***")

                successes += 1
                break

            if truncated:

                print("*** TRUNCATED ***")
                break

            obs = next_obs
            state = next_state

    env.close()

    print()
    print("Successes:", successes, "/", EVAL_EPISODES)

# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    policy_net = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    policy_net.eval()

    evaluate(policy_net)