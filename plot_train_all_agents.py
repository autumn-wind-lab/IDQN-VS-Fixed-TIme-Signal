import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

# Configuration
log_base_dir = "c:/Users/oneau/OneDrive/Desktop/Reinforcement_Learning/SUMO_TESTS/sumo-scenarios/pasubio/tensorboard_logs"
output_dir = "train_IDQN_plots"
runs = ["run0", "run1", "run2", "run3", "run4"]  
smoothing_factor = 0.99  # Smoothing factor 

# List of metrics to plot
metrics = [
    "custom/reward",
    "custom/total_reward",
    "custom/queue_length",
    "custom/wait_time",
    "custom/throughput",
    "custom/co2"
]

# Custom axis labels and titles
metric_labels = {
    "custom/reward": {"ylabel": "Reward", "title": "Reward for {}"},
    "custom/total_reward": {"ylabel": "Total Reward", "title": "Total Reward for {}"},
    "custom/queue_length": {"ylabel": "Average Queue Length (vehicles)", "title": "Average Queue Length for {}"},
    "custom/wait_time": {"ylabel": "Waiting Time (seconds)", "title": "Waiting Time for {}"},
    "custom/throughput": {"ylabel": "Cumulative Throughput (vehicles)", "title": "Throughput for {}"},
    "custom/co2": {"ylabel": "CO2 Emissions (g/vehicle)", "title": "CO2 Emissions for {}"}
}

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to apply exponential moving average (EMA) smoothing
def smooth_data(data, smoothing_factor):
    smoothed = []
    alpha = 1 - smoothing_factor
    for i, value in enumerate(data):
        if i == 0:
            smoothed.append(value)
        else:
            smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
    return smoothed

# Get list of agents from run0
run0_dir = os.path.join(log_base_dir, "run0")
if not os.path.exists(run0_dir):
    print(f"No log directory found for run0 at {run0_dir}. Please check the log directory.")
    exit()
agent_dirs = [d for d in os.listdir(run0_dir) if os.path.isdir(os.path.join(run0_dir, d))]
if not agent_dirs:
    print(f"No agent directories found in {run0_dir}. Please check the log directory.")
    exit()

# Loop over all agents
for agent_id in agent_dirs:
    print(f"Processing logs for agent: {agent_id}")

    # Plot each metric for this agent across all runs
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plotted = False  # Track if any data was plotted for this metric

        for run in runs:
            agent_log_dir = os.path.join(log_base_dir, run, agent_id)
            if not os.path.exists(agent_log_dir):
                print(f"No log directory found for {agent_id} in {run} at {agent_log_dir}")
                continue

            # Initialize EventAccumulator for the agent's log directory
            try:
                ea = event_accumulator.EventAccumulator(agent_log_dir)
                ea.Reload()  # Load all events
            except Exception as e:
                print(f"Failed to load events for {agent_id} in {run}: {e}")
                continue

            # Check if the metric exists
            available_tags = ea.Tags().get("scalars", [])
            if metric not in available_tags:
                print(f"Metric {metric} not found for {agent_id} in {run}")
                continue

            # Load and smooth data
            try:
                events = ea.Scalars(metric)
                if not events:
                    print(f"No data for {metric} in {agent_id} for {run}")
                    continue

                # Extract steps and values
                steps = [e.step for e in events]
                values = [e.value for e in events]
                smoothed_values = smooth_data(values, smoothing_factor)

                # Plot smoothed data
                plt.plot(steps, smoothed_values, label=f"Run {run.replace('run', '')}", linewidth=2)
                plotted = True
            except Exception as e:
                print(f"Error processing {metric} for {agent_id} in {run}: {e}")
                continue

        # Finalize and save the plot if data was plotted
        if plotted:
            plt.xlabel("Training Steps")
            plt.ylabel(metric_labels[metric]["ylabel"])
            plt.title(metric_labels[metric]["title"].format(agent_id))
            plt.grid(True)
            plt.legend()
            safe_metric_name = metric.replace("custom/", "").replace("/", "_")
            output_path = os.path.join(output_dir, f"{agent_id}_{safe_metric_name}.png")
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {output_path}")
        else:
            print(f"No data plotted for {metric} for {agent_id}")
            plt.close()

print("Plotting completed.")