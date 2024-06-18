#include <chrono>
#include <custom_interface.hpp>
#include <deque>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

MotorCmd array_to_motor_cmd(const pybind11::array_t<float> &input) {
    auto buf =
        input.unchecked<1>(); // Access array elements without bounds checking
    MotorCmd cmd;
    for (int i = 0; i < 12; ++i) {
        cmd.q_des[i] = buf(i);
        cmd.qd_des[i] = buf(12 + i);
        cmd.kp_des[i] = buf(24 + i);
        cmd.kd_des[i] = buf(36 + i);
        cmd.tau_des[i] = buf(48 + i);
    }
    return cmd;
}

py::array_t<float> robot_data_to_array(const RobotData &rd) {
    py::array result = py::array_t<float>(63);  // Create a new NumPy array
    auto r = result.mutable_unchecked<float>(); // Access array elements without
                                                // bounds checking
    for (int i = 0; i < 12; ++i) {
        r(i) = rd.q[i];
        r(12 + i) = rd.qd[i];
        r(24 + i) = rd.tau[i];
    }
    for (int i = 0; i < 4; ++i) {
        r(36 + i) = rd.quat[i];
    }
    for (int i = 0; i < 3; ++i) {
        r(40 + i) = rd.rpy[i];
        r(43 + i) = rd.acc[i];
        r(46 + i) = rd.omega[i];
    }
    r(49) = rd.ctrl_topic_interval;
    for (int i = 0; i < 12; ++i) {
        r(50 + i) = static_cast<float>(rd.motor_flags[i]);
    }
    r(62) = static_cast<float>(rd.err_flag);
    return result;
}

class Robot : public CustomInterface {
  public:
    Robot(const double &loop_rate, bool mode) : CustomInterface(loop_rate), mode_(mode){};
    ~Robot(){};

  private:
    long long count_ = 0;
    MotorCmd cmd_buffer_;
    bool mode_ = 0;
    float init_q_[12];
    float target1_q_[3] = {0 / 57.3, 80 / 57.3, -135 / 57.3};
    float target2_q_[3] = {0 / 57.3, 45 / 57.3, -70 / 57.3};
    float init_yaw_;
    void UserCode(bool first_run) {
        if (first_run) {
            // initially send zero frames to reset the robot
            for (int i = 0; i < 12; i++) {
                cmd_buffer_.q_des[i] = 0;
                cmd_buffer_.qd_des[i] = 0;
                cmd_buffer_.kp_des[i] = 0;
                cmd_buffer_.kd_des[i] = 0;
                cmd_buffer_.tau_des[i] = 0;
            }
            for (int i = 0; i < 12; i++)
                init_q_[i] = robot_data_.q[i];
            init_yaw_ = robot_data_.rpy[2];
            if (init_q_[2] < -0.1 && init_q_[5] < -0.1 && init_q_[8] < -0.1 &&
                init_q_[11] < -0.1)
                count_ = -1;
        } else if (mode_ == 0) {
            // mode 0: ruled based control for standing up
            float t = (count_ * dt_ / 0.8) > 2 ? 2 : (count_ * dt_ / 0.8);
            for (int i = 0; i < 12; i++) {
                if (t < 1.0)
                    cmd_buffer_.q_des[i] =
                        target1_q_[i % 3] * t + init_q_[i] * (1 - t);
                else
                    cmd_buffer_.q_des[i] = target2_q_[i % 3] * (t - 1) +
                                           target1_q_[i % 3] * (2 - t);
                cmd_buffer_.kp_des[i] = 60;
                cmd_buffer_.kd_des[i] = 2;
                cmd_buffer_.qd_des[i] = 0;
                cmd_buffer_.tau_des[i] = 0;
            }
        }

        this->motor_cmd_ = cmd_buffer_;

        if ((count_++) % 1000 == 0)
            PrintData(robot_data_);
    }

  public:
    py::array_t<float> get_robot_data() const {
        // convert robot data to numpy array with named fields to RobotData
        return robot_data_to_array(robot_data_);
    }
    float get_init_yaw() const { return init_yaw_; }
    bool get_motor_initialized() const { return motor_initialized; }

    void set_motor_cmd(const py::array_t<float> &cmd_array) {
        if (mode_ == 0) return;
        MotorCmd cmd = array_to_motor_cmd(cmd_array);
        cmd_buffer_ = cmd;
    }

    void set_mode(bool mode) { mode_ = mode; };
};

PYBIND11_MODULE(robot_interface, m) {
    py::class_<Robot, std::shared_ptr<Robot>>(m, "Robot")
        .def(py::init<const double &, bool>())
        .def("get_robot_data", &Robot::get_robot_data)
        .def("get_init_yaw", &Robot::get_init_yaw)
        .def("motor_initialized", &Robot::get_motor_initialized)
        .def("set_motor_cmd", &Robot::set_motor_cmd)
        .def("set_mode", &Robot::set_mode)
        .def("spin", &Robot::Spin)
        .def("stop", &Robot::Stop);
}
