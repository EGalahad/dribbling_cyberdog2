import numpy as np
np.set_printoptions(precision=3, suppress=True)

import os
DRIBBLE_ROOT = os.path.dirname(os.path.abspath(__file__))


# get robot and ball global position
upstream_ip = "10.0.0.144"
upstream_port = 40000

# get robot rpy
robot_server_port = 12345
robot_server_freq = 50

# get ball location relative to base
ball_detector_server_port = 12435
ball_detector_server_freq = 20
ball_detector_server_refresh_freq = 40

color = "black"
gate = {"black": np.array([2.1, 0., 0.]),
        "red": np.array([2.1, 7., 0.])}
gate_paral = {"black": np.array([0., -1., 0.]),
              "red": np.array([0., 1., 0.])}


class obs_scales:
    ang_vel = 0.25
    aux_reward_cmd = 1
    ball_pos = 1
    body_height_cmd = 2
    body_pitch_cmd = 0.3
    body_roll_cmd = 0.3
    compliance_cmd = 1
    depth_image = 1
    dof_pos = 1
    dof_vel = 0.05
    footswing_height_cmd = 0.15
    friction_measurements = 1
    gait_freq_cmd = 1
    gait_phase_cmd = 1
    height_measurements = 5
    imu = 0.1
    lin_vel = 2
    rgb_image = 1
    segmentation_image = 1
    stance_length_cmd = 1
    stance_width_cmd = 1


commands_scale = np.array(
    [
        obs_scales.lin_vel,
        obs_scales.lin_vel,
        obs_scales.ang_vel,
        obs_scales.body_height_cmd,
        obs_scales.gait_freq_cmd,
        obs_scales.gait_phase_cmd,
        obs_scales.gait_phase_cmd,
        obs_scales.gait_phase_cmd,
        obs_scales.gait_phase_cmd,
        obs_scales.footswing_height_cmd,
        obs_scales.body_pitch_cmd,
        obs_scales.body_roll_cmd,
        obs_scales.stance_width_cmd,
        obs_scales.stance_length_cmd,
        obs_scales.aux_reward_cmd,
    ]
)

class obs_scales_loco:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        imu = 0.1
        height_measurements = 5.0
        friction_measurements = 1.0
        body_height_cmd = 2.0
        gait_phase_cmd = 1.0
        gait_freq_cmd = 1.0
        footswing_height_cmd = 0.15
        body_pitch_cmd = 0.3
        body_roll_cmd = 0.3
        aux_reward_cmd = 1.0
        compliance_cmd = 1.0
        stance_width_cmd = 1.0
        stance_length_cmd = 1.0
        segmentation_image = 1.0
        rgb_image = 1.0
        depth_image = 1.0

cmds_scale_loco = np.array([
    obs_scales_loco.lin_vel, 
    obs_scales_loco.lin_vel, 
    obs_scales_loco.ang_vel,
    obs_scales_loco.body_height_cmd, 
    obs_scales_loco.gait_freq_cmd,
    obs_scales_loco.gait_phase_cmd, 
    obs_scales_loco.gait_phase_cmd,
    obs_scales_loco.gait_phase_cmd, 
    obs_scales_loco.gait_phase_cmd,
    obs_scales_loco.footswing_height_cmd, 
    obs_scales_loco.body_pitch_cmd,
    obs_scales_loco.body_roll_cmd, 
    obs_scales_loco.stance_width_cmd,
    obs_scales_loco.stance_length_cmd, 
    obs_scales_loco.aux_reward_cmd])[:15]

default_dof_pos_sim_loco = np.array([0, -70, 110] * 4) / 57.3
default_dof_pos_real_loco = np.array([0, 70, -110] * 4) / 57.3 

# 57
default_dof_pos_sim = np.array([0, -60, 100] * 2 + [0, -70, 100] * 2) / 57.3
default_dof_pos_real = np.array([0, 60, -100] * 2 + [0, 70, -100] * 2) / 57.3

# 125
default_dof_pos_sim_125 = np.array([0, -60, 100] * 2 + [0, -75, 105] * 2) / 57.3
default_dof_pos_real_125 = np.array([0, 60, -100] * 2 + [0, 75, -105] * 2) / 57.3

real2sim_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
sim2real_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

real2sim_mult = [1, -1, -1] * 4
sim2real_mult = [1, -1, -1] * 4

H = np.array([[ 0.99304481, -0.01204687, 2.24669205],
 [-0.08069741,  0.91225263, -1.27213011],
 [-0.01237622, -0.00907308,  1.        ]])


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    array = np.arange(12)
    print(array[real2sim_order][sim2real_order])

    print("default_dof_pos_sim: \n", default_dof_pos_sim)
    print(
        "default_dof_pos_real[real2sim_order] * real2sim_mult: \n",
        (default_dof_pos_real[real2sim_order] * real2sim_mult),
    )

    print("default_dof_pos_real: \n", default_dof_pos_real)
    print(
        "default_dof_pos_sim[sim2real_order] * sim2real_mult: \n",
        (default_dof_pos_sim[sim2real_order] * sim2real_mult),
    )

