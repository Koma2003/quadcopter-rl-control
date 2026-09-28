#define _USE_MATH_DEFINES
#include <cmath>
#include "quadcopter.h"
#include <iostream>
#include <algorithm>

Quadcopter::Quadcopter() : config() {
    reset();
}

Quadcopter::Quadcopter(const QuadConfig& cfg) : config(cfg) {
    reset();
}

void Quadcopter::reset() {
    x = y = z = 0.0;
    vx = vy = vz = 0.0;
    roll = pitch = yaw = 0.0;
    roll_rate = pitch_rate = yaw_rate = 0.0;
    motors = {0.0, 0.0, 0.0, 0.0};
    motor_command = {0.0, 0.0, 0.0, 0.0};
    wind_x = wind_y = wind_z = 0.0;
    sim_time = 0.0;
}

void Quadcopter::setMotors(const std::array<double, 4>& commands) {
    motor_command = commands;
}

void Quadcopter::setMotorsNormalized(const std::array<double, 4>& actions) {
    for (int i = 0; i < 4; i++) {
        double a = std::clamp(actions[i], 0.0, 1.0);
        motor_command[i] = config.motor_min + a * (config.motor_max - config.motor_min);
    }
}

void Quadcopter::setWind(double wx, double wy, double wz) {
    wind_x = wx;
    wind_y = wy;
    wind_z = wz;
}

std::array<double, 12> Quadcopter::getState() const {
    return {
        x, y, z,
        vx, vy, vz,
        roll, pitch, yaw,
        roll_rate, pitch_rate, yaw_rate
    };
}

double Quadcopter::motorThrust(int i) const {
    return config.thrust_coeff * motors[i] * motors[i];
}

void Quadcopter::updateMotors(double dt) {
    double alpha = 1.0 - std::exp(-dt / config.motor_tau);
    for (int i = 0; i < 4; i++) {
        motors[i] += alpha * (motor_command[i] - motors[i]);
        motors[i] = std::clamp(motors[i], config.motor_min, config.motor_max);
    }
}

void Quadcopter::normalizeAngles() {
    auto wrap = [](double angle) -> double {
        while (angle > M_PI)  angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    };
    roll  = wrap(roll);
    pitch = wrap(pitch);
    yaw   = wrap(yaw);
}

void Quadcopter::step(double dt) {
    updateMotors(dt);

    double T0 = motorThrust(0);
    double T1 = motorThrust(1);
    double T2 = motorThrust(2);
    double T3 = motorThrust(3);
    double T_total = T0 + T1 + T2 + T3;

    double tau_roll  = config.arm_length * (T1 - T3);
    double tau_pitch = config.arm_length * (T2 - T0);
    double tau_yaw   = config.moment_coeff * (
        motors[0] * motors[0]
      - motors[1] * motors[1]
      + motors[2] * motors[2]
      - motors[3] * motors[3]
    );

    double omega_props = motors[0] - motors[1] + motors[2] - motors[3];
    double gyro_roll  = -config.moment_coeff * pitch_rate * omega_props;
    double gyro_pitch =  config.moment_coeff * roll_rate  * omega_props;

    double roll_acc  = (tau_roll  + gyro_roll)  / config.Ixx;
    double pitch_acc = (tau_pitch + gyro_pitch) / config.Iyy;
    double yaw_acc   = tau_yaw / config.Izz;

    double damp_a = config.drag_coeff_angular;
    roll_acc  -= damp_a * roll_rate;
    pitch_acc -= damp_a * pitch_rate;
    yaw_acc   -= damp_a * yaw_rate;

    roll_rate  += roll_acc  * dt;
    pitch_rate += pitch_acc * dt;
    yaw_rate   += yaw_acc   * dt;

    roll  += roll_rate  * dt;
    pitch += pitch_rate * dt;
    yaw   += yaw_rate   * dt;

    normalizeAngles();

    double sr = std::sin(roll);
    double cr = std::cos(roll);
    double sp = std::sin(pitch);
    double cp = std::cos(pitch);
    double sy = std::sin(yaw);
    double cy = std::cos(yaw);

    double Fx_thrust = T_total * (sr * sy + cr * sp * cy);
    double Fy_thrust = T_total * (-sr * cy + cr * sp * sy);
    double Fz_thrust = T_total * (cr * cp);

    double damp_l = config.drag_coeff_linear;
    double Fx_drag = -damp_l * vx;
    double Fy_drag = -damp_l * vy;
    double Fz_drag = -damp_l * vz;

    double ax = (Fx_thrust + Fx_drag + wind_x) / config.mass;
    double ay = (Fy_thrust + Fy_drag + wind_y) / config.mass;
    double az = (Fz_thrust + Fz_drag + wind_z) / config.mass - config.gravity;

    vx += ax * dt;
    vy += ay * dt;
    vz += az * dt;

    x += vx * dt;
    y += vy * dt;
    z += vz * dt;

    if (z <= 0.0) {
        z  = 0.0;
        vz = std::max(vz, 0.0);
        vx *= 0.9;
        vy *= 0.9;
    }

    sim_time += dt;
}

bool Quadcopter::isDone() const {
    if (std::abs(roll) > config.max_angle || std::abs(pitch) > config.max_angle)
        return true;
    if (z > config.max_height)
        return true;
    double dist = std::sqrt(x * x + y * y);
    if (dist > config.max_distance)
        return true;
    return false;
}

Quadcopter::RewardInfo Quadcopter::getRewardInfo() const {
    RewardInfo info;
    info.height = z;
    info.distance_from_origin = std::sqrt(x * x + y * y);
    info.velocity_magnitude = std::sqrt(vx * vx + vy * vy + vz * vz);
    info.angle_magnitude = std::sqrt(roll * roll + pitch * pitch);
    info.angular_velocity_magnitude = std::sqrt(
        roll_rate * roll_rate + pitch_rate * pitch_rate + yaw_rate * yaw_rate
    );
    info.crashed = (std::abs(roll) > config.max_angle || std::abs(pitch) > config.max_angle);
    info.out_of_bounds = (z > config.max_height ||
        std::sqrt(x * x + y * y) > config.max_distance);
    return info;
}

void Quadcopter::printState() const {
    std::cout << "t=" << sim_time << "s\n";
    std::cout << "Pos: [" << x << ", " << y << ", " << z << "]\n";
    std::cout << "Vel: [" << vx << ", " << vy << ", " << vz << "]\n";
    std::cout << "Ang: [" << roll << ", " << pitch << ", " << yaw << "]\n";
    std::cout << "Wind: [" << wind_x << ", " << wind_y << ", " << wind_z << "]\n";
}