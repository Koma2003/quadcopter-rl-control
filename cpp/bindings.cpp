#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "quadcopter.h"

namespace py = pybind11;

PYBIND11_MODULE(quad_sim, m) {
    m.doc() = "Quadcopter simulator";

    py::class_<QuadConfig>(m, "QuadConfig")
        .def(py::init<>())
        .def_readwrite("mass",               &QuadConfig::mass)
        .def_readwrite("gravity",            &QuadConfig::gravity)
        .def_readwrite("arm_length",         &QuadConfig::arm_length)
        .def_readwrite("thrust_coeff",       &QuadConfig::thrust_coeff)
        .def_readwrite("moment_coeff",       &QuadConfig::moment_coeff)
        .def_readwrite("Ixx",                &QuadConfig::Ixx)
        .def_readwrite("Iyy",                &QuadConfig::Iyy)
        .def_readwrite("Izz",                &QuadConfig::Izz)
        .def_readwrite("motor_min",          &QuadConfig::motor_min)
        .def_readwrite("motor_max",          &QuadConfig::motor_max)
        .def_readwrite("motor_tau",          &QuadConfig::motor_tau)
        .def_readwrite("drag_coeff_linear",  &QuadConfig::drag_coeff_linear)
        .def_readwrite("drag_coeff_angular", &QuadConfig::drag_coeff_angular)
        .def_readwrite("max_angle",          &QuadConfig::max_angle)
        .def_readwrite("max_height",         &QuadConfig::max_height)
        .def_readwrite("max_distance",       &QuadConfig::max_distance);

    py::class_<Quadcopter::RewardInfo>(m, "RewardInfo")
        .def_readonly("height",                    &Quadcopter::RewardInfo::height)
        .def_readonly("distance_from_origin",      &Quadcopter::RewardInfo::distance_from_origin)
        .def_readonly("velocity_magnitude",        &Quadcopter::RewardInfo::velocity_magnitude)
        .def_readonly("angle_magnitude",           &Quadcopter::RewardInfo::angle_magnitude)
        .def_readonly("angular_velocity_magnitude",&Quadcopter::RewardInfo::angular_velocity_magnitude)
        .def_readonly("crashed",                   &Quadcopter::RewardInfo::crashed)
        .def_readonly("out_of_bounds",             &Quadcopter::RewardInfo::out_of_bounds);

    py::class_<Quadcopter>(m, "Quadcopter")
        .def(py::init<>())
        .def(py::init<const QuadConfig&>())
        .def_readwrite("x",          &Quadcopter::x)
        .def_readwrite("y",          &Quadcopter::y)
        .def_readwrite("z",          &Quadcopter::z)
        .def_readwrite("vx",         &Quadcopter::vx)
        .def_readwrite("vy",         &Quadcopter::vy)
        .def_readwrite("vz",         &Quadcopter::vz)
        .def_readwrite("roll",       &Quadcopter::roll)
        .def_readwrite("pitch",      &Quadcopter::pitch)
        .def_readwrite("yaw",        &Quadcopter::yaw)
        .def_readwrite("roll_rate",  &Quadcopter::roll_rate)
        .def_readwrite("pitch_rate", &Quadcopter::pitch_rate)
        .def_readwrite("yaw_rate",   &Quadcopter::yaw_rate)
        .def_readwrite("sim_time",   &Quadcopter::sim_time)
        .def_readwrite("wind_x",     &Quadcopter::wind_x)
        .def_readwrite("wind_y",     &Quadcopter::wind_y)
        .def_readwrite("wind_z",     &Quadcopter::wind_z)
        .def("reset",                &Quadcopter::reset)
        .def("step",                 &Quadcopter::step)
        .def("get_state",            &Quadcopter::getState)
        .def("set_motors",           &Quadcopter::setMotors)
        .def("set_motors_normalized",&Quadcopter::setMotorsNormalized)
        .def("set_wind",             &Quadcopter::setWind)
        .def("is_done",              &Quadcopter::isDone)
        .def("get_reward_info",      &Quadcopter::getRewardInfo)
        .def("print_state",          &Quadcopter::printState);
}