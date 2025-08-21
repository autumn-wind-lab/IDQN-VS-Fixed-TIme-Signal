import gymnasium as gym
from stable_baselines3 import DQN
from parallel_pasubio_env_rewardv5 import PasubioTrafficEnv
import numpy as np
from stable_baselines3.common.logger import configure
import traci
import time
import torch
from multiprocessing import Process
import os
import psutil

class SB3SpacesOnly(gym.Env):
    def __init__(self, action_space, observation_space):
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space
    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}
    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

def train_run(run_id, port, sumocfg_file, log_dir, max_steps, episodes):
    try:
        print(f"Starting run {run_id} on port {port}")
        env = PasubioTrafficEnv(
            sumocfg_file=sumocfg_file,
            port=port,
            run_id=f"run{run_id}",
            max_steps=max_steps,
            log_file=f"training_metrics_run{run_id}.txt",
            use_gui=False
        )
        print(f"Run {run_id} on port {port}: Agents in environment: {list(env.agents)}")

        # Build one DQN per agent using a 'spaces-only' dummy env
        agents = {}
        run_log_dir = f"{log_dir}/run{run_id}"
        os.makedirs(run_log_dir, exist_ok=True)

        TRAIN_FREQ = 8
        LEARNING_STARTS = 500
        BATCH_SIZE = 256

        for agent in env.agents:
            logger = configure(f"{run_log_dir}/{agent}", ["tensorboard"])
            dummy_env = SB3SpacesOnly(env.action_spaces[agent], env.observation_spaces[agent])
            # Define a deeper network with three hidden layers
            policy_kwargs = dict(net_arch=[128, 128, 128])  # Three layers with 128 units each
            agents[agent] = DQN(
                policy="MlpPolicy",
                env=dummy_env,
                learning_rate=0.0005,
                buffer_size=100000,
                learning_starts=LEARNING_STARTS,
                batch_size=BATCH_SIZE,
                tau=0.005,
                gamma=0.99,
                train_freq=TRAIN_FREQ,
                gradient_steps=1,
                target_update_interval=500,
                exploration_fraction=0.3,
                exploration_final_eps=0.05,
                verbose=1,
                tensorboard_log=f"{run_log_dir}/{agent}",
                device="cuda",
                policy_kwargs=policy_kwargs  
            )
            agents[agent].set_logger(logger)

        # Caches for logging
        last_reward = {a: 0.0 for a in env.agents}
        last_infer = {a: 0.0 for a in env.agents}
        last_obs = {a: None for a in env.agents}
        steps_since_train = {a: 0 for a in env.agents}

        for episode in range(episodes):
            observations, infos = env.reset()
            total_rewards = {agent: 0.0 for agent in env.agents}
            total_queue_lengths = {agent: [] for agent in env.agents}
            episode_start_time = time.time()
            step = 0

            print(f"Run {run_id}, Episode {episode + 1}/{episodes}")

            while step < max_steps:
                # Get actions for all agents
                actions = {}
                for agent in env.agents:
                    obs = observations[agent]
                    last_obs[agent] = obs
                    action, _ = agents[agent].predict(obs, deterministic=False)
                    actions[agent] = action

                # Step the environment with all actions
                observations, rewards, terminations, truncations, infos = env.step(actions)

                # Update caches and replay buffers
                for agent in env.agents:
                    obs = last_obs[agent]
                    next_obs = observations[agent]
                    reward = rewards[agent]
                    action = actions[agent]
                    done_flag = bool(terminations[agent] or truncations[agent])

                    last_reward[agent] = reward
                    last_infer[agent] = infos[agent].get("inference_time", 0.0)
                    total_rewards[agent] += reward
                    total_queue_lengths[agent].append(obs[0])

                    # Add transition to agent's replay buffer
                    agents[agent].replay_buffer.add(
                        obs=obs,
                        next_obs=next_obs,
                        action=np.array([action]),
                        reward=np.array([reward], dtype=np.float32),
                        done=np.array([done_flag], dtype=np.float32),
                        infos=[{}]
                    )
                    steps_since_train[agent] += 1

                    # Train when train_freq reached and after learning_starts
                    total_env_steps_for_agent = agents[agent].replay_buffer.size()
                    if steps_since_train[agent] >= TRAIN_FREQ and total_env_steps_for_agent >= LEARNING_STARTS:
                        agents[agent].train(gradient_steps=1)
                        steps_since_train[agent] = 0

                # TensorBoard logging every 100 steps
                if step % 100 == 0:
                    vehicles_remaining = traci.simulation.getMinExpectedNumber()
                    vehicles = traci.vehicle.getIDList()
                    num_vehicles = max(1, len(vehicles))
                    current_time = traci.simulation.getTime()
                    travel_time = sum(
                        current_time - traci.vehicle.getDeparture(v)
                        for v in vehicles if traci.vehicle.getDeparture(v) >= 0
                    ) / num_vehicles
                    co2 = sum(traci.vehicle.getCO2Emission(v) for v in vehicles) / (num_vehicles * 1000)

                    for agent in env.agents:
                        log_obs = last_obs[agent]
                        queue_length = log_obs[0] * env.max_queue
                        wait_time = np.mean([
                            traci.lane.getWaitingTime(lane)
                            for lane in traci.trafficlight.getControlledLanes(agent)
                        ]) if traci.trafficlight.getControlledLanes(agent) else 0.0
                        #norm_queue_length = log_obs[0]
                        #norm_wait_time = wait_time / env.max_waiting_time
                        # norm_travel_time = travel_time / env.max_travel_time
                        # norm_throughput = env._cumulative_throughput / env.max_throughput
                        # norm_co2 = co2 / env.max_co2

                        unique_step = episode * max_steps + step
                        agents[agent].logger.record("custom/reward", last_reward[agent])
                        agents[agent].logger.record("custom/action", actions[agent])
                        agents[agent].logger.record("custom/total_reward", total_rewards[agent])
                        agents[agent].logger.record("custom/queue_length", queue_length)
                        agents[agent].logger.record("custom/wait_time", wait_time)
                        agents[agent].logger.record("custom/throughput", env._cumulative_throughput)
                        agents[agent].logger.record("custom/vehicles_remaining", vehicles_remaining)
                        agents[agent].logger.record("custom/inference_time", last_infer[agent])
                        #agents[agent].logger.record("custom/norm_queue_length", norm_queue_length)
                        #agents[agent].logger.record("custom/norm_wait_time", norm_wait_time)
                        #agents[agent].logger.record("custom/norm_travel_time", norm_travel_time)
                        #agents[agent].logger.record("custom/norm_throughput", norm_throughput)
                        #agents[agent].logger.record("custom/norm_co2", norm_co2)
                        agents[agent].logger.record("custom/co2", co2)
                        agents[agent].logger.dump(step=unique_step)

                # Periodic console print
                if step % 600 == 0:
                    print(
                        f"Run {run_id}, Ep {episode + 1}, Step {step + 1}/{max_steps}, "
                        f"Rewards: {{ {', '.join(f'{k}: {v:.3f}' for k,v in rewards.items())} }}, "
                        f"Total: {{ {', '.join(f'{k}: {v:.2f}' for k,v in total_rewards.items())} }}"
                    )

                # Episode end
                if all(terminations.values()) or all(truncations.values()):
                    print(f"Run {run_id}, Episode {episode + 1} terminated at step {step + 1}: "
                          f"Terminated={all(terminations.values())}, Truncated={all(truncations.values())}")
                    break

                step += 1

            # End-of-episode logging
            episode_time = time.time() - episode_start_time
            for idx, agent in enumerate(env.agents):
                unique_step = episode * max_steps + max_steps + idx
                agents[agent].logger.record("custom/episode_time", episode_time)
                agents[agent].logger.dump(step=unique_step)

            print(
                f"Run {run_id}, Episode {episode + 1}/{episodes} Completed, "
                f"Total Rewards: {{ {', '.join(f'{k}: {round(v,2)}' for k,v in total_rewards.items())} }}, "
                f"Avg Queue Lengths: {{ {', '.join(f'{k}: {round(np.mean(v) * env.max_queue,2) if len(v)>0 else 0.0}' for k,v in total_queue_lengths.items())} }}, "
                f"Episode Time: {episode_time:.2f}s"
            )

        os.makedirs(f"dqn_models/run{run_id}", exist_ok=True)
        for agent in env.agents:
            agents[agent].save(f"dqn_models/run{run_id}/dqn_model_{agent}")

    except Exception as e:
        print(f"Run {run_id} failed on port {port}: {e}")
        with open(f"training_metrics_run{run_id}.txt", "a") as f:
            f.write(f"Run {run_id} failed on port {port}: {e}\n")
    finally:
        env.close()

if __name__ == "__main__":
    # Verify CUDA
    print(f"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Configuration
    SUMOCFG_FILE = "c:/Users/oneau/OneDrive/Desktop/Reinforcement_Learning/SUMO_TESTS/sumo-scenarios/pasubio/run.sumocfg"
    LOG_DIR = "c:/Users/oneau/OneDrive/Desktop/Reinforcement_Learning/SUMO_TESTS/sumo-scenarios/pasubio/tensorboard_logs"

    # Configuration
    MAX_STEPS = 1800  
    EPISODES = 500
    PORTS = [8813, 8814, 8815, 8816, 8817]

    # Clean up lingering SUMO processes
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] in ['sumo.exe', 'sumo-gui.exe']:
            proc.kill()

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("dqn_models", exist_ok=True)

    processes = []
    for i, port in enumerate(PORTS):
        p = Process(target=train_run, args=(i, port, SUMOCFG_FILE, LOG_DIR, MAX_STEPS, EPISODES))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()