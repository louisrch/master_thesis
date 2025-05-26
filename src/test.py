import metaworld
import random
from PIL import Image
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.35,
    "azimuth": 160,
    "elevation": -35.0,
    "lookat": np.array([0, 0.6, -0.05]),
}

name = "drawer-close-v3"
ml1 = metaworld.ML1(name)
env = ml1.train_classes[name](render_mode="rgb_array", camera_name="chatGPTCam")
# or

#print(help(env.mujoco_renderer.render))
#env.mujoco_renderer = MujocoRenderer(env.model, env.data, DEFAULT_CAMERA_CONFIG, width=400, height=400)
task = random.choice(ml1.train_tasks)
env.set_task(task)
obs = env.reset()

# Use the camera name as defined in your XML!
r = env.render()
img = Image.fromarray(r)
img.save("assets/" + name + ".png")


# env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
# task = random.choice(ml1.train_tasks)
# env.set_task(task)  # Set task

# obs = env.reset()  # Reset environment
# a = env.action_space.sample()  # Sample an action
# obs, reward, done, info, _ = env.step(a)  # Step the environment with the sampled random action

keys = {'assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-insert-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-v2', 'push-wall-v2', 'push-back-v2', 'reach-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2'}
