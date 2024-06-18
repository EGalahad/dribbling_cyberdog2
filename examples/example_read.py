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


robot = robot_interface.Robot(50, 1)

time.sleep(0.1)
print("init yaw:", robot.get_init_yaw())
np.set_printoptions(precision=2, suppress=True)
try:
    while True:
        robot_data = parse_robot_data(robot.get_robot_data())
        # yaw = robot_data["rpy"][0][2]
        # print(yaw)
        print(robot_data["q"][0])
        time.sleep(1)
except KeyboardInterrupt:
    robot.stop()
