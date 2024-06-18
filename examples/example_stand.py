"""Example code to demonstrate how to use the robot_interface module.
"""
import robot_interface
import time
import numpy as np


def parse_robot_data(arr: np.ndarray):
    # Define the dtype for the structured array
    dtype = np.dtype([
        ("q", np.float32, (12,)),
        ("qd", np.float32, (12,)),
        ("tau", np.float32, (12,)),
        ("quat", np.float32, (4,)),
        ("rpy", np.float32, (3,)),
        ("acc", np.float32, (3,)),
        ("omega", np.float32, (3,)),
        ("ctrl_topic_interval", np.float32),
        ("motor_flags", np.float32, (12,)),
        ("err_flag", np.float32)
    ])
    
    # Create a zero-initialized array of the structured dtype with a single entry
    structured_array = np.zeros(1, dtype=dtype)
    
    # Fill the structured array with data from the flat array
    structured_array['q'] = arr[0:12]
    structured_array['qd'] = arr[12:24]
    structured_array['tau'] = arr[24:36]
    structured_array['quat'] = arr[36:40]
    structured_array['rpy'] = arr[40:43]
    structured_array['acc'] = arr[43:46]
    structured_array['omega'] = arr[46:49]
    structured_array['ctrl_topic_interval'] = arr[49]
    structured_array['motor_flags'] = arr[50:62]
    structured_array['err_flag'] = arr[62]
    
    return structured_array


# scale = 1
# kp, kd = 60, 2
control_freq = 500


# scale = 2
# kp, kd = 60, 2
# control_freq = 200


def stand(robot, kp=60, kd=2, completion_time=2.0):
    cmd = np.zeros(60, dtype=np.float32)
    cmd[24:36] = kp
    cmd[36:48] = kd

    first_run = True
    st = None
    init_q = None

    target1_q = np.array([ 10 / 57.3, 80 / 57.3, -135 / 57.3 ] * 4, dtype=np.float32)
    target2_q = np.array([ 10 / 57.3, 45 / 57.3, -70 / 57.3 ] * 4, dtype=np.float32)

    scale = completion_time / 2.0

    while True:
        t = time.time()
        if first_run:
            robot_data = robot.get_robot_data()
            robot_data = parse_robot_data(robot_data)
            init_q = robot_data["q"]
            if np.isnan(init_q).any():
                print("Initial q has nan, skip this run.")
                continue
            first_run = False
            st = t
        # 1
        robot_data = robot.get_robot_data()
        robot_data = parse_robot_data(robot_data)
        t = t - st
        if t > 2 * scale:
            t = 2 * scale
        if t < scale:
            target_q = target1_q * t / scale + init_q * (1 - t / scale)
        else:
            target_q = target2_q * (t / scale - 1) + target1_q * (2 - t / scale)
        # 2
        # target_q = init_q

        cmd[:12] = target_q
        robot.set_motor_cmd(cmd)
        time.sleep(max(1 / 500 + t - time.time(), 0))



robot = robot_interface.Robot(control_freq, 1)
stand(robot)
