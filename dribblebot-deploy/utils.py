import numpy as np
import time
import sys


class Timer:
    def __init__(self, name) -> None:
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.name} time: {self.elapsed_time:.4f} seconds")


def project_gravity(quaternion: np.ndarray):
    w, x, y, z = quaternion  # assume normalized
    gx = 2 * w * y - 2 * x * z
    gy = -2 * w * x - 2 * y * z
    gz = -w * w + x * x + y * y - z * z
    return np.array([gx, gy, gz])


def rotate2d(vec, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ vec

def inverse_rotate_rpy(vec, rpy):
    r, p, y = rpy
    
    # Rotation matrix around z-axis (yaw)
    c, s = np.cos(y), np.sin(y)
    R_yaw = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])

    # Rotation matrix around y-axis (pitch)
    c, s = np.cos(p), np.sin(p)
    R_pitch = np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])

    # Rotation matrix around x-axis (roll)
    c, s = np.cos(r), np.sin(r)
    R_roll = np.array([[1, 0, 0],
                       [0, c, -s],
                       [0, s, c]])

    # Combine the rotation matrices: roll -> pitch -> yaw
    R = R_roll @ R_pitch @ R_yaw

    # Inverse rotation matrix
    R_inv = np.linalg.inv(R)

    # Apply the inverse rotation to the vector
    return R_inv @ vec

def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


robot_data_dtype = np.dtype(
    [
        ("q", np.float32, (12,)),
        ("qd", np.float32, (12,)),
        ("tau", np.float32, (12,)),
        ("quat", np.float32, (4,)),
        ("rpy", np.float32, (3,)),
        ("acc", np.float32, (3,)),
        ("omega", np.float32, (3,)),
        ("ctrl_topic_interval", np.float32),
        ("motor_flags", np.float32, (12,)),
        ("err_flag", np.float32),
    ]
)

def parse_robot_data(arr: np.ndarray):
    # Define the dtype for the structured array
    # Create a zero-initialized array of the structured dtype with a single entry
    structured_array = np.zeros(1, dtype=robot_data_dtype)

    # Fill the structured array with data from the flat array
    structured_array["q"] = arr[0:12]
    structured_array["qd"] = arr[12:24]
    structured_array["tau"] = arr[24:36]
    structured_array["quat"] = arr[36:40]
    structured_array["rpy"] = arr[40:43]
    structured_array["acc"] = arr[43:46]
    structured_array["omega"] = arr[46:49]
    structured_array["ctrl_topic_interval"] = arr[49]
    structured_array["motor_flags"] = arr[50:62]
    structured_array["err_flag"] = arr[62]

    return structured_array


def stand(robot, kp=60, kd=2, completion_time=2.0, target_q=None):
    import time

    cmd = np.zeros(60, dtype=np.float32)
    cmd[24:36] = kp
    cmd[36:48] = kd

    first_run = True
    st = None
    init_q = None

    target1_q = np.array([10 / 57.3, 80 / 57.3, -135 / 57.3] * 4, dtype=np.float32)
    if target_q is not None:
        target2_q = target_q
    else:
        target2_q = np.array([10 / 57.3, 45 / 57.3, -70 / 57.3] * 4, dtype=np.float32)
        

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
            break
        if t < scale:
            target_q = target1_q * t / scale + init_q * (1 - t / scale)
        else:
            target_q = target2_q * (t / scale - 1) + target1_q * (2 - t / scale)
        # 2
        # target_q = init_q

        cmd[:12] = target_q
        robot.set_motor_cmd(cmd)
        time.sleep(max(1 / 500 + t - time.time(), 0))

def wait_for_enter():
    print("Execution paused. Press Enter to continue...")
    while True:
        time.sleep(0.01)
        if sys.stdin.read(1) == '\n':
            break

