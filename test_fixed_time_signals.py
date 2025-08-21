import gymnasium as gym
from parallel_pasubio_env_rewardv5 import PasubioTrafficEnv
import numpy as np
import traci
import time
import os
import psutil
from multiprocessing import Process
import csv
import socket
from collections import defaultdict

# Configuration
SUMOCFG_FILE = "c:/Users/oneau/OneDrive/Desktop/Reinforcement_Learning/SUMO_TESTS/sumo-scenarios/pasubio/test.sumocfg"
TEST_DIR = "c:/Users/oneau/OneDrive/Desktop/Reinforcement_Learning/SUMO_TESTS/sumo-scenarios/pasubio/test_metrics_fixed_time"
MAX_STEPS = 1800
EPISODES = 20
PORTS = [8813, 8814, 8815, 8816, 8817]

def test_run(run_id, port, sumocfg_file, test_dir, max_steps, episodes):
    env = None
    try:
        print(f"Starting test run {run_id} on port {port}")
        
        # Verify port availability
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                print(f"Port {port} is free for run {run_id}")
            except socket.error:
                raise ValueError(f"Port {port} is already in use for run {run_id}")

        # Initialize environment
        env = PasubioTrafficEnv(
            sumocfg_file=sumocfg_file,
            port=port,
            run_id=f"test_run{run_id}",
            max_steps=max_steps,
            log_file=f"{test_dir}/test_log_run{run_id}.txt",
            use_gui=False
        )

        # Create directories for metrics
        os.makedirs(f"{test_dir}/local_metrics", exist_ok=True)
        os.makedirs(f"{test_dir}/global_metrics", exist_ok=True)
        os.makedirs(f"{test_dir}/agentspecific_global_metrics", exist_ok=True)

        # Open CSV files
        local_metrics_file = f"{test_dir}/local_metrics/run{run_id}_local_metrics.csv"
        global_metrics_file = f"{test_dir}/global_metrics/run{run_id}_global_metrics.csv"
        agent_global_metrics_file = f"{test_dir}/agentspecific_global_metrics/run{run_id}_agentspecific_global_metrics.csv"

        with open(local_metrics_file, "a", newline="") as f_local, \
             open(global_metrics_file, "a", newline="") as f_global, \
             open(agent_global_metrics_file, "a", newline="") as f_agent_global:

            # Write headers if files are empty
            if os.path.getsize(local_metrics_file) == 0:
                writer_local = csv.writer(f_local)
                writer_local.writerow(["episode", "algorithm", "time_step", "agent_id", "avg_queue_length", "avg_waiting_time"])
            if os.path.getsize(global_metrics_file) == 0:
                writer_global = csv.writer(f_global)
                writer_global.writerow([
                    "episode", "algorithm", "total_throughput",
                    "avg_speed", "stops_per_vehicle", "total_CO2",
                    "vehicles_remaining", "avg_queue_length"
                ])
            if os.path.getsize(agent_global_metrics_file) == 0:
                writer_agent_global = csv.writer(f_agent_global)
                writer_agent_global.writerow(["episode", "algorithm", "agent_id", "avg_waiting_time", "avg_queue_length"])

            writer_local = csv.writer(f_local)
            writer_global = csv.writer(f_global)
            writer_agent_global = csv.writer(f_agent_global)

            for episode in range(episodes):
                # Calculate seed for this episode: run_id * episodes + episode + 1
                seed = run_id * episodes + episode + 1
                print(f"Test Run {run_id}, Episode {episode + 1}/{episodes}, Seed: {seed}")
                observations, infos = env.reset(seed=seed)
                print(f"Test Run {run_id}, Episode {episode + 1}, SUMO Seed: {traci.simulation.getParameter('', 'seed')}, Initial Vehicles: {len(traci.vehicle.getIDList())}")

                # Initialize accumulators for global metrics
                total_throughput = 0
                speed_per_step = []
                co2_per_step = []
                queue_lengths_per_agent = {agent: [] for agent in env.agents}
                waiting_times_per_agent = {agent: [] for agent in env.agents}
                vehicle_stops = defaultdict(int)
                vehicle_prev_stopped = {}  
                current_time = 0.0

                for step in range(max_steps):
                    # Update current_time
                    current_time = traci.simulation.getTime()

                    # Update vehicle stops
                    vehicles = traci.vehicle.getIDList()
                    for v in vehicles:
                        if traci.vehicle.isStopped(v):
                            if not vehicle_prev_stopped.get(v, False):
                                vehicle_stops[v] += 1
                        vehicle_prev_stopped[v] = traci.vehicle.isStopped(v)

                    # Step environment with no actions (use fixed-time signals)
                    observations, rewards, terminations, truncations, infos = env.step({})

                    # Log current phase for each traffic light to verify fixed-time signals
                    for agent in env.agents:
                        current_phase = traci.trafficlight.getPhase(agent)
                        print(f"Test Run {run_id}, Episode {episode + 1}, Step {step + 1}, Agent {agent}, Current Phase: {current_phase}")

                    # Collect local metrics
                    for agent in env.agents:
                        lanes = traci.trafficlight.getControlledLanes(agent)
                        avg_queue_length = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes) / max(1, len(lanes))
                        avg_waiting_time = sum(traci.lane.getWaitingTime(l) for l in lanes) / max(1, len(lanes))

                        # Log local metrics
                        writer_local.writerow([episode + 1, "FixedTime", step + 1, agent, avg_queue_length, avg_waiting_time])

                        # Accumulate for agent-specific global metrics
                        queue_lengths_per_agent[agent].append(avg_queue_length)
                        waiting_times_per_agent[agent].append(avg_waiting_time)

                    # Collect global metrics
                    total_throughput += traci.simulation.getArrivedNumber()
                    vehicles = traci.vehicle.getIDList()
                    num_vehicles = max(1, len(vehicles))
                    avg_speed = sum(traci.vehicle.getSpeed(v) for v in vehicles) / num_vehicles
                    total_co2 = sum(traci.vehicle.getCO2Emission(v) for v in vehicles) / 1000.0

                    speed_per_step.append(avg_speed)
                    co2_per_step.append(total_co2)

                    # Flush local metrics
                    f_local.flush()

                    if all(terminations.values()) or all(truncations.values()):
                        print(f"Test Run {run_id}, Episode {episode + 1} terminated at step {step + 1}")
                        break

                # Calculate global metrics for the episode
                avg_speed = float(np.mean(speed_per_step)) if speed_per_step else 0.0
                avg_stops_per_vehicle = (
                    sum(vehicle_stops.values()) / len(vehicle_stops) if vehicle_stops else 0.0
                )
                avg_co2 = float(np.mean(co2_per_step)) if co2_per_step else 0.0
                vehicles_remaining = traci.simulation.getMinExpectedNumber()

                # Average queue length across all agents (episode-level)
                all_queue_means = [np.mean(vals) for vals in queue_lengths_per_agent.values() if len(vals) > 0]
                global_avg_queue_length = float(np.mean(all_queue_means)) if all_queue_means else 0.0

                # Log global metrics
                print(f"Writing global metrics for run {run_id} to {global_metrics_file}")
                writer_global.writerow([
                    episode + 1, "FixedTime", total_throughput,
                    avg_speed, avg_stops_per_vehicle, avg_co2, vehicles_remaining,
                    global_avg_queue_length
                ])
                f_global.flush()

                # Log agent-specific global metrics
                for agent in env.agents:
                    agent_avg_wait = float(np.mean(waiting_times_per_agent[agent])) if waiting_times_per_agent[agent] else 0.0
                    agent_avg_queue = float(np.mean(queue_lengths_per_agent[agent])) if queue_lengths_per_agent[agent] else 0.0
                    writer_agent_global.writerow([episode + 1, "FixedTime", agent, agent_avg_wait, agent_avg_queue])
                f_agent_global.flush()

                print(
                    f"Test Run {run_id}, Episode {episode + 1} Completed: "
                    f"Total Throughput: {total_throughput}, "
                    f"Avg Queue Length: {global_avg_queue_length:.2f}, "
                    f"Avg Stops per Vehicle: {avg_stops_per_vehicle:.3f}"
                )

    except Exception as e:
        print(f"Test Run {run_id} failed on port {port}: {e}")
        try:
            os.makedirs(test_dir, exist_ok=True)
            with open(f"{test_dir}/run{run_id}_error.log", "a") as f:
                f.write(f"Test Run {run_id} failed on port {port}: {e}\n")
        except Exception:
            pass
    finally:
        if env is not None:
            env.close()

if __name__ == "__main__":
    # Clean up lingering SUMO processes
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] in ['sumo.exe', 'sumo-gui.exe']:
            proc.kill()

    # Create test directory
    os.makedirs(TEST_DIR, exist_ok=True)

    # Start parallel test runs
    processes = []
    for i, port in enumerate(PORTS):
        p = Process(
            target=test_run,
            args=(i, port, SUMOCFG_FILE, TEST_DIR, MAX_STEPS, EPISODES)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()