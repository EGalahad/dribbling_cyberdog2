import robot_interface
from utils import *
from config import *
from ball_detector import BallDetector

import numpy as np
import time

st = time.time()
import torch
print(f"import torch takes: {time.time() - st} seconds")


class DribbleEnv:
    # observation parameters
    obs_dim = 75
    privi_obs_dim = 6
    act_dim = 12
    history_len = 15

    # gait type parameters
    phase = 0.5
    offset = 0.0
    bound = 0.0
    foot_gait_offsets = [phase + offset + bound, offset, bound, phase]

    # commands
    duration = 0.5  # duration = stance / (stance + swing)
    step_frequency = 3.0

    # this is substeps in dribblebot and walk these ways
    control_decimation = 4
    simulation_dt = 0.005
    dt = control_decimation * simulation_dt

    action_scale = 0.25
    hip_scale_reduction = 1.0
    # kp = 20
    # kd = 0.5
    # fix: this works better but mismatch sim
    kp = 60
    kd = 2

    yaw_init = 0

    commands = np.array(
        [
            0,  # x vel
            0,  # y vel
            0.0,  # yaw vel
            0.0,  # body height
            step_frequency,
            phase,
            offset,
            bound,
            duration,
            0.09,  # foot swing height
            0.0,  # pitch
            0.0,  # roll
            0.0,  # stance_width
            0.1 / 2,  # stance length
            0.01 / 2,  # unknown
        ]
    )

    def __init__(self, robot: robot_interface.Robot, ball_detector: BallDetector):
        self.robot = robot
        self.ball_detector = ball_detector
        self.buffer = torch.zeros(
            self.history_len * 3, self.obs_dim, dtype=torch.float32
        )
        self.t = self.history_len

        self.action_t = torch.zeros(self.act_dim, dtype=torch.float32)
        self.action_t_minus1 = torch.zeros(self.act_dim, dtype=torch.float32)

        self.gait_index = 0.0

        self.motor_cmd = np.zeros(60, dtype=np.float32)
        self.motor_cmd[24:36] = self.kp
        self.motor_cmd[36:48] = self.kd

    def observe(self, commands: np.ndarray):  # commands : [x_vel, y_vel]
        # self.ball_detector.refresh()
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
        ball_pos, robot_pos, oppenent_pos, mode = self.ball_detector.get_pos()
        # clip ball pos norm to 1
        clip_norm = 1
        ball_pos_norm = np.linalg.norm(ball_pos[:2])
        clip_scale = max(1, ball_pos_norm / clip_norm)
        if clip_scale > 1:
            ball_pos[:2] = ball_pos[:2] / clip_scale
        projected_gravity = project_gravity(robot_obs["quat"][0])
        self.commands[:2] = commands
        commands = self.commands * commands_scale

        dof_pos = (
            robot_obs["q"][0][real2sim_order] * real2sim_mult - default_dof_pos_sim
        ) * obs_scales.dof_pos
        dof_vel = (
            robot_obs["qd"][0][real2sim_order] * real2sim_mult * obs_scales.dof_vel
        )
        action = self.action_t
        last_action = self.action_t_minus1

        clock = self.clock()
        yaw = wrap_to_pi(robot_obs["rpy"][0][2] - self.yaw_init)
        yaw = wrap_to_pi(robot_obs["rpy"][0][2] - self.yaw_init)
        timing = self.gait_index

        return torch.cat(
            [
                torch.as_tensor(ball_pos),
                torch.as_tensor(projected_gravity),
                torch.as_tensor(commands),
                torch.as_tensor(dof_pos),
                torch.as_tensor(dof_vel),
                action,
                last_action,
                torch.as_tensor(clock),
                torch.as_tensor([yaw]),
                torch.as_tensor([timing]),
            ]
        )

    def clock(self):
        return [
            np.sin(2 * np.pi * (self.gait_index + offset))
            for offset in self.foot_gait_offsets
        ]

    def step(self, action):
        self.action_t_minus1[:] = self.action_t
        self.action_t[:] = action

        action_scaled = (
            action.detach().numpy() * self.action_scale * self.hip_scale_reduction
        )
        dof_pos_target = action_scaled[-1] + default_dof_pos_sim
        self.motor_cmd[:12] = dof_pos_target[sim2real_order] * sim2real_mult
        self.robot.set_motor_cmd(self.motor_cmd)

        self.gait_index = (self.gait_index + self.step_frequency * self.dt) % 1

def load_poliy():
    run_name = "helpful-river-57"
    iter = "latest"
    
    body_file = f"{DRIBBLE_ROOT}/ckpt/{run_name}/body_{iter}.jit"
    adaptation_file = (
        f"{DRIBBLE_ROOT}/ckpt/{run_name}/adaptation_module_{iter}.jit"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = torch.load(body_file, map_location=device)
    adaptation_module = torch.load(adaptation_file, map_location=device)

    body = body.half()
    adaptation_module = adaptation_module.half()

    def policy(obs, info={}):
        obs = obs.half().to(device)
        latent = adaptation_module(obs)
        action = body(torch.cat((obs, latent), dim=-1))
        # info["latent"] = latent
        return action.cpu()
    return policy

if __name__ == "__main__":
    control_freq = 50
    decimation = 1
    mode = 1
    robot = robot_interface.Robot(control_freq, mode)
    st = time.time()
    while not robot.motor_initialized():
        time.sleep(1)
    print(f"Waited {time.time() - st:.3f} seconds for motor intialize!")

    stand_completion_time = 1.0
    stand(robot, kp=60, kd=2, completion_time=stand_completion_time, target_q=default_dof_pos_real)

    ball_detector = BallDetector(port=ball_detector_server_port)
    env = DribbleEnv(robot=robot, ball_detector=ball_detector)

    policy = load_poliy()
    # warm up policy
    obs, robot_obs = env.observe(np.array([0.0, -0.5]))
    for i in range(100):
        obs = obs.reshape(1, -1)
        action = policy(obs)

    env.yaw_init = robot_obs["rpy"][0, 2]

    interval = 1 / control_freq * decimation
    try:
        with torch.inference_mode():
            while True:
                # with Timer("dribble iter"):
                    begin = time.perf_counter()
                    command_ball_vel = np.array([0.5, 0.0])
                    obs, robot_obs = env.observe(command_ball_vel)
                    obs = obs.reshape(1, -1)
                    action = policy(obs)
                    env.step(action)
                    end = time.perf_counter()
                    time.sleep(max(0, begin + interval - end))
    except KeyboardInterrupt:
        robot.stop()
