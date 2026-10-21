import gymnasium
import ale_py

gymnasium.register_envs(ale_py)

env = gymnasium.make("ALE/Pong-v5", render_mode="rgb_array")
env = gymnasium.wrappers.RecordVideo(
    env,
    episode_trigger=lambda num: num % 2 == 0,
    video_folder="saved-video-folder",
    name_prefix="video-",
)
for episode in range(10):
    obs, info = env.reset()
    episode_over = False

    while not episode_over:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated


# from pacman_dqn import make_env, get_dots_eaten

# env = make_env(seed=0)
# env.reset()
# previous = get_dots_eaten(env)

# for step in range(5000):
#     obs, reward, done, truncated, _ = env.step(env.action_space.sample() % 5)
#     dots = get_dots_eaten(env)
#     if reward != 0 or dots != previous:
#         print(step, "reward", reward, "ram[119]", dots)
#     previous = dots
#     if done or truncated:
#         break



env.close()