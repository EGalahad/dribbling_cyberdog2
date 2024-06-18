import time

st = time.time()
import torch

print("import torch takes: ", time.time() - st)
import numpy as np
from utils import *
from config import *
from ball_detector import BallDetector
import robot_interface


class LocoEnv:
    num_obs = 70
    history_len = 30
    num_actions = 12

    duration = 0.5
    step_freq = 3.0

    gait = torch.tensor([0.5, 0.0, 0.0])

    decimation = 4
    simulation_dt = 0.005
    dt = decimation * simulation_dt
    freq = 3.0

    action_scale = 0.25
    hip_scale_reduction = 1.0
    clip_actions = 5.0

    # num_commands should have 15 dim.
    commands = torch.tensor(
        [
            0.5,  # x vel
            0.0,  # y vel
            0.0,  # yaw vel
            0.2,  # body height
            2.0,  # step freq
            0.5,
            0.0,
            0.0,  # phase params
            0.5,  # duration
            0.08,  # swing height
            0.0,  # pitch cmd
            0.0,  # roll cmd
            0.25,  # stance width
            0.0,  # unknown
            0.0,  # unknown
        ]
    )

    def __init__(self, robot: robot_interface.Robot):
        self.robot = robot
        self.buffer = torch.zeros(
            self.history_len * 3, self.num_obs, dtype=torch.float32
        )
        self.t = self.history_len
        self.action_t = torch.zeros(self.num_actions, dtype=torch.float32)
        self.action_t_minus1 = torch.zeros(self.num_actions, dtype=torch.float32)

        self.gait_indices = 0
        self.foot_indices = torch.zeros(4, dtype=torch.float32)
        self.clock_inputs = torch.zeros(4, dtype=torch.float32)

        self.motor_cmd = np.zeros(60, dtype=np.float32)
        self.motor_cmd[24:36] = 50.0  # kp
        self.motor_cmd[36:48] = 1  # kd

    def observe(self, commands: torch.Tensor):  # commands : [x_vel, y_vel, yaw_vel]
        robot_obs = parse_robot_data(self.robot.get_robot_data())
        obs = self.make_obs(robot_obs, commands)
        self.store_obs(obs)
        return self.buffer[self.t - self.history_len : self.t], robot_obs

    def store_obs(self, obs: torch.Tensor):
        h, buffer, t = self.history_len, self.buffer, self.t
        if t == buffer.shape[0]:
            buffer[:h] = buffer[t - h : t].clone()
            t = h
        buffer[t] = obs
        self.t = t + 1

    def make_obs(self, robot_obs: np.ndarray, commands: np.ndarray) -> torch.Tensor:
        projected_gravity = project_gravity(robot_obs["quat"][0])
        dof_pos = (
            robot_obs["q"][0][real2sim_order] * real2sim_mult - default_dof_pos_sim_loco
        ) * obs_scales_loco.dof_pos
        dof_vel = (
            robot_obs["qd"][0][real2sim_order] * real2sim_mult
        ) * obs_scales_loco.dof_vel
        action = self.action_t
        last_action = self.action_t_minus1
        self.commands[:2] = commands

        phases = self.commands[5]
        offsets = self.commands[6]
        bounds = self.commands[7]
        foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases,
        ]
        self.foot_indices = torch.remainder(torch.tensor(foot_indices), 1.0)
        self.clock_inputs = torch.sin(2 * torch.pi * self.foot_indices)

        observation = torch.cat(
            (
                torch.tensor(projected_gravity),
                self.commands * torch.tensor(cmds_scale_loco),
                torch.tensor(dof_pos),
                torch.tensor(dof_vel),
                action,
                last_action,
                self.clock_inputs,
            ),
            dim=0,
        )

        return observation

    def step(self, action: torch.Tensor):
        self.action_t_minus1 = self.action_t
        # clip action first
        self.action_t = torch.clip(action[0], -self.clip_actions, self.clip_actions)

        action_scaled = (
            self.action_t.detach().numpy()
            * self.action_scale
            * self.hip_scale_reduction
        )
        dof_pos_target = action_scaled + default_dof_pos_sim_loco
        self.motor_cmd[:12] = dof_pos_target[sim2real_order] * sim2real_mult
        self.robot.set_motor_cmd(self.motor_cmd)

        self.gait_indices = (self.gait_indices + self.freq * self.dt) % 1


if __name__ == "__main__":
    import time

    # load env
    control_freq = 50
    decimation = 1
    robot = robot_interface.Robot(control_freq, 1)
    env = LocoEnv(robot=robot)

    stand_completion_time = 1.0
    stand(
        robot,
        kp=60,
        kd=2,
        completion_time=stand_completion_time,
        target_q=default_dof_pos_real,
    )

    # load policy
    run_name = "202405030_2"  # add privilege rewards
    body_file = f"{DRIBBLE_ROOT}/ckpt/{run_name}/body_latest.jit"
    adaptation_file = f"{DRIBBLE_ROOT}/ckpt/{run_name}/adaptation_module_latest.jit"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = torch.jit.load(body_file, map_location=device)
    adaptation_module = torch.jit.load(adaptation_file, map_location=device)
    body = body.half()
    adaptation_module = adaptation_module.half()

    def policy(obs, info={}):
        obs = obs.half().to(device)
        latent = adaptation_module(obs)
        action = body(torch.cat((obs, latent), dim=-1))
        # info["latent"] = latent
        return action.cpu()

    # warm up policy
    obs, robot_obs = env.observe(torch.tensor([1.0, 0.0]))
    for i in range(100):
        obs = obs.reshape(1, -1)
        action = policy(obs)

    env.yaw_init = robot_obs["rpy"][0][2]

    interval = 1 / control_freq * decimation
    # policy inference & env step loop
    try:
        with torch.inference_mode():
            while True:
                begin = time.perf_counter()
                command_test = torch.tensor([-1.0, 0.0])
                obs, robot_obs = env.observe(command_test)
                obs = obs.reshape(1, -1)
                begin_policy = time.perf_counter()
                action = policy(obs)
                # assert action.shape == torch.Size([1, 12])
                env.step(action)
                end = time.perf_counter()
                time.sleep(max(0, begin + interval - end))
    except KeyboardInterrupt:
        robot.stop()
