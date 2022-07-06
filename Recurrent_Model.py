import matplotlib.pyplot as plt
import gym
import os
import time
import random
import time
import numpy as np
from skimage import transform
from IPython.display import display, clear_output
import tensorflow as tf
from utils import FrameStack, compute_returns, compute_gae
from ppo_rnn import PPO
from vec_env.subproc_vec_env import SubprocVecEnv
import cv2

env_name = "CarRacing-v0"

def preprocess_frame(frame):
    frame = frame[:-12, 6:-6] # Crop to 84x84
    frame = np.dot(frame[..., 0:3], [0.299, 0.587, 0.114])
    frame = frame / 255.0
    frame = frame * 2 - 1
    return frame

def make_env():
    return gym.make(env_name)

def evaluate(model, test_env, discount_factor, frame_stack_size, make_video=False):
    total_reward = 0
    test_env.seed(0)
    initial_frame = test_env.reset()
    frame_stack = FrameStack(initial_frame, stack_size=frame_stack_size, preprocess_fn=preprocess_frame)
    rendered_frame = test_env.render(mode="rgb_array")
    values, rewards, dones = [], [], []
    features = np.zeros((1, *model.get_feature_shape()))
    if make_video:
        video_writer = cv2.VideoWriter(os.path.join(model.video_dir, "step{}.avi".format(model.step_idx)),
                                       cv2.VideoWriter_fourcc(*"MPEG"), 30,
                                       (rendered_frame.shape[1], rendered_frame.shape[0]))
    while True:
        # Predict action given state: π(a_t | s_t; θ)
        state = frame_stack.get_state()
        action, value, features = model.predict(np.expand_dims(state, axis=0), features, greedy=False)
        frame, reward, done, _ = test_env.step(action[0])
        rendered_frame = test_env.render(mode="rgb_array")
        total_reward += reward
        dones.append(done)
        values.append(value)
        rewards.append(reward)
        frame_stack.add_frame(frame)
        if make_video: video_writer.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
        if done: break
    if make_video: video_writer.release()
    returns = compute_returns(np.transpose([rewards], [1, 0]), [0], np.transpose([dones], [1, 0]), discount_factor)
    value_error = np.mean(np.square(np.array(values) - returns))
    return total_reward, value_error

def train():
    # Create test env
    print("Creating test environment")
    test_env = gym.make(env_name)

    # Traning parameters
    lr_scheduler    = lambda step_idx: 3e-4 * 0.85 ** (step_idx // 10000)
    discount_factor = 0.99 # gamma
    gae_lambda      = 0.95 # lambda
    ppo_epsilon     = 0.2  # epsilon
    t_max           = 128  # T
    num_epochs      = 10   # K
    batch_size      = 128  # M
    save_interval   = 1000
    eval_interval   = 200
    training        = True

    # Environment constants
    num_envs         = 8 # N
    frame_stack_size = 1
    input_shape      = (84, 84, frame_stack_size)
    num_actions      = test_env.action_space.shape[0]
    action_min       = np.array([-1.0, 0.0, 0.0])
    action_max       = np.array([ 1.0, 1.0, 1.0])

    # Create model
    print("Creating model")
    model = PPO(num_actions, input_shape, None, action_min, action_max, ppo_epsilon,
                value_scale=0.5, entropy_scale=0.01,
                model_name="CarRacing-v0-rnn-v2-test_final2")

    if training:
        print("Creating environments")
        envs = SubprocVecEnv([make_env for _ in range(num_envs)])

        initial_frames = envs.reset()
        envs.get_images()
        frame_stacks = [FrameStack(initial_frames[i], stack_size=frame_stack_size, preprocess_fn=preprocess_frame) for i in range(num_envs)]
        features_t = [np.zeros(model.get_feature_shape()) for _ in range(num_envs)]
        
        print("Training loop")
        while True:
            # While there are running environments
            states, prev_features, taken_actions, values, rewards, dones = [], [], [], [], [], []
            
            # Simulate game for some number of steps
            for _ in range(t_max):
                # Predict and value action given state
                # π(a_t | s_t; θ_old)
                prev_features.append(features_t) # [T, N, 2592]
                states_t = [frame_stacks[i].get_state() for i in range(num_envs)]
                actions_t, values_t, features_t = model.predict(states_t,
                                                                features_t,
                                                                use_old_policy=True)

                # Sample action from a Gaussian distribution
                envs.step_async(actions_t)
                frames_t, rewards_t, dones_t, _ = envs.step_wait()
                envs.get_images() # render
                
                
                # Store state, action and reward
                states.append(states_t)                      # [T, N, 84, 84, 1]
                taken_actions.append(actions_t)              # [T, N, 3]
                values.append(np.squeeze(values_t, axis=-1)) # [T, N]
                rewards.append(rewards_t)                    # [T, N]
                dones.append(dones_t)                        # [T, N]

                # Get new state
                for i in range(num_envs):
                    # Reset environment's recurrent feature if done
                    if dones_t[i]:
                        for _ in range(frame_stack_size):
                            frame_stacks[i].add_frame(frames_t[i])
                        features_t[i] = np.zeros(model.get_feature_shape())
                    else:
                        frame_stacks[i].add_frame(frames_t[i])

            # Calculate last values (bootstrap values)
            states_last = [frame_stacks[i].get_state() for i in range(num_envs)]
            last_values = np.squeeze(model.predict(states_last,
                                                   features_t,
                                                   use_old_policy=True)[1], axis=-1) # [N]

            # Compute returns
            returns = compute_returns(rewards, last_values, dones, discount_factor)
            
            # Compute advantages
            advantages = compute_gae(rewards, values, last_values, dones, discount_factor, gae_lambda)

            # Normalize advantages
            advantages = (advantages - np.mean(advantages)) / np.std(advantages)

            # Flatten arrays
            states        = np.array(states).reshape((-1, *input_shape))       # [T x N, 84, 84, 1]
            prev_features = np.array(prev_features).reshape((-1, *model.get_feature_shape()))
            taken_actions = np.array(taken_actions).reshape((-1, num_actions)) # [T x N, 3]
            returns       = returns.flatten()                                  # [T x N]
            advantages    = advantages.flatten()                               # [T X N]

            # Train for some number of epochs
            model.update_old_policy() # θ_old <- θ
            for _ in range(num_epochs):
                # Evaluate model
                if model.step_idx % eval_interval == 0:
                    print("Running evaluation...")
                    avg_reward, value_error = evaluate(model, test_env, discount_factor, frame_stack_size, make_video=True)
                    model.write_to_summary("eval_avg_reward", avg_reward)
                    model.write_to_summary("eval_value_error", value_error)
                    
                # Save model
                if model.step_idx % save_interval == 0:
                    model.save()

                # Sample mini-batch randomly and train
                mb_idx = np.random.choice(len(states), batch_size, replace=False)

                # Optimize network
                print("Training (step {})...".format(model.step_idx))
                model.train(states[mb_idx], prev_features[mb_idx], taken_actions[mb_idx],
                            returns[mb_idx], advantages[mb_idx],
                            learning_rate=lr_scheduler)

    # Training complete, evaluate model
    avg_reward = evaluate(model, test_env, 10)
    print("Model achieved a final reward of:", avg_reward)

if __name__ == "__main__":
    train()