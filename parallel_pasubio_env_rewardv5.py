from pettingzoo import AECEnv
import traci
import sumolib
import numpy as np
from gymnasium import spaces
import os
import socket
import time

class PasubioTrafficEnv(AECEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        sumocfg_file,
        port=8813,
        run_id="run0",
        max_steps=1800,
        max_queue=50,
        max_waiting_time=2000,
        min_green_time=10,
        log_file=None,
        use_gui=False,
    ):
        super().__init__()
        self.sumocfg_file = sumocfg_file
        self.port = port
        self.run_id = run_id
        self.max_steps = max_steps
        self.max_queue = max_queue
        self.max_waiting_time = max_waiting_time
        self.min_green_time = min_green_time
        self.log_file = log_file.replace(".txt", f"_{run_id}.txt") if log_file else None
        self.log_file_handle = open(self.log_file, "a") if self.log_file else None
        self.use_gui = use_gui
        self.label = f"conn_{run_id}"
        self._cumulative_throughput = 0

        sumo_binary = "sumo-gui" if use_gui else "sumo"
        sumo_binary_path = sumolib.checkBinary(sumo_binary)

        if not self._is_port_free(self.port):
            raise ValueError(f"Port {self.port} is already in use for {self.run_id}")

        sumo_cmd = [
            sumo_binary_path,
            "-c",
            self.sumocfg_file,
            "--output-prefix",
            f"{self.run_id}.",
            "--log",
            f"sumo_log_{self.run_id}.txt",
            "--no-step-log",
            "--start",
        ]
        traci.start(sumo_cmd, port=self.port, label=self.label, numRetries=5)

        # Initialize agents
        self.agents = sorted(traci.trafficlight.getIDList())
        if not self.agents:
            raise ValueError(f"No traffic lights found in {self.sumocfg_file}")
        self.n_agents = len(self.agents)

        # Per-agent action and observation spaces
        self.action_spaces = {}
        for agent in self.agents:
            n_phases = len(traci.trafficlight.getAllProgramLogics(agent)[0].phases)
            self.action_spaces[agent] = spaces.Discrete(n_phases)

        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
            for agent in self.agents
        }

        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.current_step = 0

        # Track last phase change time + previous queues for delta reward
        self.last_change_time = {agent: 0 for agent in self.agents}
        self.prev_queue_length = {agent: 0.0 for agent in self.agents}

        if self.log_file_handle:
            self.log_file_handle.write(
                "sim_time,agent_id,queue_len,wait_time,throughput,cum_throughput,reward,inference_time,error\n"
            )
            self.log_file_handle.flush()

    def _is_port_free(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return True
            except socket.error:
                return False

    def reset(self, seed=None, options=None):
        try:
            traci.switch(self.label)
            traci.close()
        except traci.exceptions.TraCIException:
            pass

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_binary_path = sumolib.checkBinary(sumo_binary)

        if not self._is_port_free(self.port):
            raise ValueError(
                f"Port {self.port} is already in use for {self.run_id} during reset"
            )

        sumo_cmd = [
            sumo_binary_path,
            "-c",
            self.sumocfg_file,
            "--output-prefix",
            f"{self.run_id}.",
            "--log",
            f"sumo_log_{self.run_id}.txt",
            "--no-step-log",
            "--start",
        ]
        if seed is not None:
            sumo_cmd.extend(["--seed", str(seed)])
        traci.start(sumo_cmd, port=self.port, label=self.label, numRetries=5)

        self.current_step = 0
        self._cumulative_throughput = 0
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.last_change_time = {agent: 0 for agent in self.agents}
        self.prev_queue_length = {agent: 0.0 for agent in self.agents}

        if self.log_file_handle:
            self.log_file_handle.write(f"\n--- New Episode ({self.run_id}) ---\n")
            self.log_file_handle.flush()

        observations = {agent: self._observe(agent) for agent in self.agents}
        return observations, self.infos

    def _observe(self, agent):
        traci.switch(self.label)
        lanes = traci.trafficlight.getControlledLanes(agent)
        queue_length = (
            sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes) / max(1, len(lanes))
        )
        norm_queue_length = min(queue_length / self.max_queue, 1.0)
        phases = traci.trafficlight.getAllProgramLogics(agent)[0].phases
        current_phase = traci.trafficlight.getPhase(agent) / max(1, len(phases) - 1)
        return np.array([norm_queue_length, current_phase], dtype=np.float32)

    def step(self, actions):
        step_start = time.time()
        traci.switch(self.label)

        # Apply actions per agent with min-green-time logic
        for agent, action in actions.items():
            n_phases = len(traci.trafficlight.getAllProgramLogics(agent)[0].phases)
            chosen_phase = int(np.clip(action, 0, n_phases - 1))

            if self.current_step - self.last_change_time[agent] >= self.min_green_time:
                traci.trafficlight.setPhase(agent, chosen_phase)
                self.last_change_time[agent] = self.current_step

        traci.simulationStep()
        self.current_step += 1
        inference_time = time.time() - step_start

        # Compute rewards
        for agent in self.agents:
            self.rewards[agent] = self._get_individual_reward(agent)
            self.infos[agent]["inference_time"] = inference_time

        # Logging
        if self.log_file_handle:
            traci.switch(self.label)
            step_throughput = traci.simulation.getArrivedNumber()
            self._cumulative_throughput += step_throughput
            current_time = traci.simulation.getTime()
            for agent in self.agents:
                lanes = traci.trafficlight.getControlledLanes(agent)
                q_a = (
                    sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
                    / max(1, len(lanes))
                )
                w_a = (
                    sum(traci.lane.getWaitingTime(l) for l in lanes) / max(1, len(lanes))
                )
                r_a = self.rewards.get(agent, 0.0)
                inf_a = self.infos.get(agent, {}).get("inference_time", 0.0)
                err_a = self.infos.get(agent, {}).get("error", "")
                self.log_file_handle.write(
                    f"{current_time:.2f},{agent},{q_a:.4f},{w_a:.4f},{step_throughput:.4f},{self._cumulative_throughput:.4f},{r_a:.4f},{inf_a:.4f},{err_a}\n"
                )
            self.log_file_handle.flush()

        # Check termination and truncation
        terminated = (traci.simulation.getMinExpectedNumber() == 0) and (
            self.current_step >= self.max_steps
        )
        truncated = self.current_step >= self.max_steps
        for agent in self.agents:
            self.terminations[agent] = terminated
            self.truncations[agent] = truncated

        observations = {agent: self._observe(agent) for agent in self.agents}
        return observations, self.rewards, self.terminations, self.truncations, self.infos

    def _get_individual_reward(self, agent):
        traci.switch(self.label)
        lanes = traci.trafficlight.getControlledLanes(agent)
        queue_length = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes) / max(1, len(lanes))
        reward = -queue_length
        return reward

    def render(self):
        if not self.use_gui:
            print("Rendering requires use_gui=True.")

    def close(self):
        if self.log_file_handle:
            self.log_file_handle.close()
        try:
            traci.switch(self.label)
            traci.close()
        except traci.exceptions.TraCIException:
            pass