# Multi-Agent Reinforcement Learning for Optimized Traffic Signal Control

This repository contains Python scripts for simulating and analyzing traffic control strategies using the SUMO (Simulation of Urban MObility) traffic simulator. The project implements and compares two traffic signal control approaches: Fixed-Time control and Independent Deep Q-Network (IDQN) reinforcement learning. It includes scripts for running simulations, training IDQN models, and plotting metrics for performance comparison.

## Project Structure

The project consists of several Python scripts designed to work together to simulate traffic scenarios, train reinforcement learning models, and visualize results.

### Files

- **parallel_pasubio_env_rewardv5.py**: Defines the `PasubioTrafficEnv` class, a PettingZoo-based environment interfacing with SUMO for multi-agent traffic simulation. It manages traffic lights as agents, defines observation and action spaces, and computes rewards based on queue length.
- **parallel_train_cuda_v2.py**: Trains Independent DQN (IDQN) models for each traffic light agent in parallel using Stable Baselines3. It logs training metrics to TensorBoard and saves trained models.
- **test_fixed_time_signals.py**: Runs simulations using fixed-time traffic signal control, collects local and global metrics, and saves them to CSV files for analysis.
- **test_dqn_models_v1.py**: Tests pre-trained IDQN models by running simulations, collecting local and global metrics, and saving them to CSV files.
- **plot_train_all_agents.py**: Generates plots from TensorBoard logs for training metrics (e.g., reward, queue length, CO2 emissions) across multiple runs and agents.
- **plot_testmetrics_comparison.py**: Creates comparison plots (line and bar charts) for local, agent-specific, and global metrics between Fixed-Time and IDQN control strategies.

## Prerequisites

To run the scripts, you need the following dependencies installed:

- Python 3.8+
- SUMO (Simulation of Urban MObility) 1.8.0 or higher
- Required Python packages:
  - `gymnasium`
  - `stable-baselines3`
  - `numpy`
  - `traci`
  - `sumolib`
  - `psutil`
  - `torch` (with CUDA support for GPU acceleration, optional)
  - `pandas`
  - `matplotlib`
  - `tensorboard`
  - `pettingzoo`

## Installation

1. **Install SUMO**:
   - Download and install SUMO from [eclipse.org/sumo](https://www.eclipse.org/sumo/).
   - Ensure the SUMO binary (`sumo` or `sumo-gui`) is accessible in your system PATH.

2. **Install Python dependencies**:
   ```bash
   pip install gymnasium stable-baselines3 numpy traci sumolib psutil torch pandas matplotlib tensorboard pettingzoo
   ```

3. **Clone this repository**:
   ```bash
   git clone https://github.com/autumn-wind-lab/IDQN-VS-Fixed-TIme-Signal
   cd IDQN-VS-Fixed-TIme-Signal
   ```

4. **Prepare the SUMO configuration file**:
   - Ensure the `test.sumocfg` and `run.sumocfg` files are present and update the paths in the scripts (`SUMOCFG_FILE` variables) to point to your SUMO configuration files.

## Usage

### 1. Training IDQN Models
To train IDQN models for traffic light control:

```bash
python parallel_train_cuda_v2.py
```

- This script runs parallel training for multiple runs (default: 5 runs) on specified ports (8813–8817).
- Training metrics are logged to `tensorboard_logs/runX` directories.
- Trained models are saved to `dqn_models/runX`.

### 2. Testing Fixed-Time Control
To run simulations with fixed-time traffic signal control:

```bash
python test_fixed_time_signals.py
```

- Outputs metrics to `test_metrics_fixed_time/{local_metrics,global_metrics,agentspecific_global_metrics}` directories.

### 3. Testing IDQN Models
To test pre-trained IDQN models:

```bash
python test_dqn_models_v1.py
```

- Ensure trained models are available in the `dqn_models/run1` directory.
- Outputs metrics to `test_metrics_dqn/{local_metrics,global_metrics,agentspecific_global_metrics}` directories.

### 4. Plotting Training Metrics
To generate plots for training metrics:

```bash
python plot_train_all_agents.py
```

- Generates plots for each agent and metric in the `train_IDQN_plots` directory.

### 5. Comparing Test Metrics
To compare Fixed-Time and IDQN performance:

```bash
python plot_testmetrics_comparison.py
```

- Generates comparison plots in `test_IDQN_&_FixedTime_plots/{local_metrics,agent_specific_global_metrics,global_metrics}` directories.

## Configuration

- **SUMO Configuration**: Update `SUMOCFG_FILE` in `parallel_train_cuda_v2.py`, `test_fixed_time_signals.py`, and `test_dqn_models_v1.py` to point to your `.sumocfg` file.
- **Ports**: The scripts use ports 8813–8817 for parallel SUMO instances. Ensure these ports are free or modify the `PORTS` list in the scripts.
- **Output Directories**:
  - Training logs: `tensorboard_logs`
  - Model storage: `dqn_models`
  - Test metrics: `test_metrics_dqn`, `test_metrics_fixed_time`
  - Plots: `train_IDQN_plots`, `test_IDQN_&_FixedTime_plots`
- **Simulation Parameters**:
  - `MAX_STEPS`: 1800 (simulation steps per episode)
  - `EPISODES`: 500 for training, 20 for testing
  - `NUM_RUNS`: 5 (parallel runs)

## Notes

- Ensure no other SUMO instances are running before starting simulations, as the scripts terminate lingering SUMO processes (`sumo.exe`, `sumo-gui.exe`).
- For GPU acceleration, ensure CUDA is installed and PyTorch is configured with CUDA support.
- The `plot_testmetrics_comparison.py` script assumes metrics from both Fixed-Time and IDQN tests are available in their respective directories.
- If you encounter port conflicts, check for processes using ports 8813–8817 and terminate them or modify the `PORTS` list.


## Acknowledgments

- Built using [SUMO](https://www.eclipse.org/sumo/) for traffic simulation.
- Utilizes [Stable Baselines3](https://stable-baselines3.readthedocs.io/) for reinforcement learning.
- Leverages [PettingZoo](https://pettingzoo.farama.org/) for multi-agent environments.
