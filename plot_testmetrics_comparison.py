import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define directories for metrics and output
DQN_DIR = "C:/Users/oneau/OneDrive/Desktop/Reinforcement_Learning/SUMO_TESTS/sumo-scenarios/pasubio/test_metrics_dqn"
FIXED_TIME_DIR = "C:/Users/oneau/OneDrive/Desktop/Reinforcement_Learning/SUMO_TESTS/sumo-scenarios/pasubio/test_metrics_fixed_time"
OUTPUT_DIR_LOCAL = "C:/Users/oneau/OneDrive/Desktop/Reinforcement_Learning/SUMO_TESTS/sumo-scenarios/pasubio/test_IDQN_&_FixedTime_plots/local_metrics"
OUTPUT_DIR_AGENT_SPECIFIC = "C:/Users/oneau/OneDrive/Desktop/Reinforcement_Learning/SUMO_TESTS/sumo-scenarios/pasubio/test_IDQN_&_FixedTime_plots/agent_specific_global_metrics"
OUTPUT_DIR_GLOBAL = "C:/Users/oneau/OneDrive/Desktop/Reinforcement_Learning/SUMO_TESTS/sumo-scenarios/pasubio/test_IDQN_&_FixedTime_plots/global_metrics"
MAX_STEPS = 1800
NUM_RUNS = 5
NUM_EPISODES = 20

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR_LOCAL, exist_ok=True)
os.makedirs(OUTPUT_DIR_AGENT_SPECIFIC, exist_ok=True)
os.makedirs(OUTPUT_DIR_GLOBAL, exist_ok=True)

def load_metrics(base_dir, folder, run_id, file_prefix):
    """Load metrics from CSV for a given run and folder."""
    file_path = os.path.join(base_dir, folder, f"run{run_id}_{file_prefix}_metrics.csv")
    if not os.path.exists(file_path):
        print(f"Metrics file not found: {file_path}")
        return None
    df = pd.read_csv(file_path)
    return df

def process_local_metrics(base_dir, metric_name, algorithm):
    """Process local metrics (time-series) for all runs and episodes."""
    run_data = []
    for run_id in range(NUM_RUNS):
        df = load_metrics(base_dir, "local_metrics", run_id, "local")
        if df is None:
            continue

        # Filter for the metric and group by episode and time_step
        metric_data = df[["episode", "time_step", metric_name]]
        # Compute mean across episodes for each time step
        episode_means = metric_data.groupby(["episode", "time_step"])[metric_name].mean().unstack()
        # Average across episodes for this run
        run_mean = episode_means.mean(axis=0).reindex(range(1, MAX_STEPS + 1), fill_value=0.0)
        run_data.append(run_mean.values)

    if not run_data:
        print(f"No data found for {algorithm} local metric {metric_name}")
        return None, None

    # Compute mean and std across runs
    run_data = np.array(run_data)
    mean_across_runs = np.mean(run_data, axis=0)
    std_across_runs = np.std(run_data, axis=0)
    return mean_across_runs, std_across_runs

def process_agent_specific_global_metrics(base_dir, metric_name, algorithm):
    """Process agent-specific global metrics for all runs and episodes."""
    run_data_by_agent = {}
    for run_id in range(NUM_RUNS):
        df = load_metrics(base_dir, "agentspecific_global_metrics", run_id, "agentspecific_global")
        if df is None:
            continue

        # Group by agent_id and episode
        for agent in df["agent_id"].unique():
            agent_data = df[df["agent_id"] == agent][["episode", metric_name]]
            # Compute mean across episodes for this agent
            agent_mean = agent_data.groupby("episode")[metric_name].mean().mean()
            if agent not in run_data_by_agent:
                run_data_by_agent[agent] = []
            run_data_by_agent[agent].append(agent_mean)

    if not run_data_by_agent:
        print(f"No data found for {algorithm} agent-specific global metric {metric_name}")
        return None

    # Compute mean and std across runs for each agent
    result = {}
    for agent in run_data_by_agent:
        run_data = np.array(run_data_by_agent[agent])
        result[agent] = {
            "mean": np.mean(run_data),
            "std": np.std(run_data)
        }
    return result

def process_global_metrics(base_dir, metric_name, algorithm):
    """Process global metrics for all runs and episodes."""
    run_data = []
    for run_id in range(NUM_RUNS):
        df = load_metrics(base_dir, "global_metrics", run_id, "global")
        if df is None:
            continue

        # Compute mean across episodes
        metric_mean = df[metric_name].mean()
        run_data.append(metric_mean)

    if not run_data:
        print(f"No data found for {algorithm} global metric {metric_name}")
        return None, None

    # Compute mean and std across runs
    run_data = np.array(run_data)
    mean_across_runs = np.mean(run_data)
    std_across_runs = np.std(run_data)
    return mean_across_runs, std_across_runs

def plot_local_metrics_comparison(dqn_mean, dqn_std, fixed_mean, fixed_std, metric_name):
    """Plot comparison of local metrics for DQN and Fixed-Time."""
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(1, MAX_STEPS + 1)

    # Plot DQN
    plt.plot(time_steps, dqn_mean, label="IDQN (Mean)", color="blue")
    plt.fill_between(
        time_steps,
        dqn_mean - dqn_std,
        dqn_mean + dqn_std,
        color="blue",
        alpha=0.2,
        label="IDQN (Std Dev)"
    )

    # Plot Fixed-Time
    plt.plot(time_steps, fixed_mean, label="Fixed-Time (Mean)", color="red")
    plt.fill_between(
        time_steps,
        fixed_mean - fixed_std,
        fixed_mean + fixed_std,
        color="red",
        alpha=0.2,
        label="Fixed-Time (Std Dev)"
    )

    plt.xlabel("Time Step")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"Comparison of {metric_name.replace('_', ' ').title()}: IDQN vs Fixed-Time")
    plt.grid(True)
    plt.legend()
    output_path = os.path.join(OUTPUT_DIR_LOCAL, f"{metric_name}_comparison.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_agent_specific_global_metrics_comparison(dqn_data, fixed_data, metric_name):
    """Plot comparison of agent-specific global metrics for DQN and Fixed-Time as bar charts."""
    if dqn_data is None or fixed_data is None:
        print(f"Cannot plot agent-specific global metric {metric_name} due to missing data")
        return

    agents = set(dqn_data.keys()) | set(fixed_data.keys())
    for agent in agents:
        plt.figure(figsize=(6, 6))
        dqn_mean = dqn_data.get(agent, {"mean": 0, "std": 0})["mean"]
        fixed_mean = fixed_data.get(agent, {"mean": 0, "std": 0})["mean"]

        # Plot bars without error bars
        plt.bar(
            ["IDQN", "Fixed-Time"],
            [dqn_mean, fixed_mean],
            color=["blue", "red"]
        )

        plt.ylabel(metric_name.replace("_", " ").title())
        plt.title(f"Comparison of {metric_name.replace('_', ' ').title()} for {agent}: IDQN vs Fixed-Time")
        plt.grid(True, axis="y")
        output_path = os.path.join(OUTPUT_DIR_AGENT_SPECIFIC, f"{agent}_{metric_name}_comparison.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {output_path}")

def plot_global_metrics_comparison(dqn_mean, dqn_std, fixed_mean, fixed_std, metric_name):
    """Plot comparison of global metrics for DQN and Fixed-Time as bar charts."""
    if dqn_mean is None or fixed_mean is None:
        print(f"Cannot plot global metric {metric_name} due to missing data")
        return

    plt.figure(figsize=(6, 6))
    # Plot bars without error bars
    plt.bar(
        ["IDQN", "Fixed-Time"],
        [dqn_mean, fixed_mean],
        color=["blue", "red"]
    )

    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"Comparison of {metric_name.replace('_', ' ').title()}: IDQN vs Fixed-Time")
    plt.grid(True, axis="y")
    output_path = os.path.join(OUTPUT_DIR_GLOBAL, f"{metric_name}_comparison.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")

def main():
    # Local metrics
    local_metrics = ["avg_queue_length", "avg_waiting_time"]
    for metric in local_metrics:
        dqn_mean, dqn_std = process_local_metrics(DQN_DIR, metric, "IDQN")
        fixed_mean, fixed_std = process_local_metrics(FIXED_TIME_DIR, metric, "FixedTime")
        if dqn_mean is not None and fixed_mean is not None:
            plot_local_metrics_comparison(dqn_mean, dqn_std, fixed_mean, fixed_std, metric)
        else:
            print(f"Skipping plot for local metric {metric} due to missing data")

    # Agent-specific global metrics
    agent_specific_metrics = ["avg_queue_length", "avg_waiting_time"]
    for metric in agent_specific_metrics:
        dqn_data = process_agent_specific_global_metrics(DQN_DIR, metric, "IDQN")
        fixed_data = process_agent_specific_global_metrics(FIXED_TIME_DIR, metric, "FixedTime")
        plot_agent_specific_global_metrics_comparison(dqn_data, fixed_data, metric)

    # Global metrics
    global_metrics = [
        "total_throughput", "avg_speed",
        "stops_per_vehicle", "total_CO2", "vehicles_remaining", "avg_queue_length"
    ]
    for metric in global_metrics:
        dqn_mean, dqn_std = process_global_metrics(DQN_DIR, metric, "IDQN")
        fixed_mean, fixed_std = process_global_metrics(FIXED_TIME_DIR, metric, "FixedTime")
        plot_global_metrics_comparison(dqn_mean, dqn_std, fixed_mean, fixed_std, metric)

    print_config = [
        {"type": "Local Metrics", "metrics": local_metrics, "path": OUTPUT_DIR_LOCAL},
        {"type": "Agent-Specific Global Metrics", "metrics": agent_specific_metrics, "path": OUTPUT_DIR_AGENT_SPECIFIC},
        {"type": "Global Metrics", "metrics": global_metrics, "path": OUTPUT_DIR_GLOBAL}
    ]
    
    for config in print_config:
        print(f"\n{config['type']} saved to: {config['path']}")
        print(f"Metrics processed: {', '.join(config['metrics'])}")
    
    print("\nPlotting completed.")

if __name__ == "__main__":
    main()