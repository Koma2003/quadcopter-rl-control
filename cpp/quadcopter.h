#pragma once

#include <array>
#include <string>

struct QuadConfig {
    double mass          = 1.0;
    double gravity       = 9.81;
    double arm_length    = 0.2;

    double thrust_coeff  = 1.0e-5;
    double moment_coeff  = 2.0e-7;

    double Ixx = 0.02;
    double Iyy = 0.02;
    double Izz = 0.04;

    double motor_min     = 0.0;
    double motor_max     = 1000.0;
    double motor_tau     = 0.05;

    double drag_coeff_linear  = 0.1;
    double drag_coeff_angular = 0.05;

    double max_angle     = 1.2;
    double max_height    = 50.0;
    double max_distance  = 50.0;
};

class Quadcopter {
public:
    double x, y, z;
    double vx, vy, vz;
    double roll, pitch, yaw;
    double roll_rate, pitch_rate, yaw_rate;

    std::array<double, 4> motors;
    std::array<double, 4> motor_command;

    double wind_x, wind_y, wind_z;

    QuadConfig config;
    double sim_time;

    Quadcopter();
    Quadcopter(const QuadConfig& cfg);

    void reset();
    void setMotors(const std::array<double, 4>& commands);
    void setMotorsNormalized(const std::array<double, 4>& actions);
    void setWind(double wx, double wy, double wz);

    std::array<double, 12> getState() const;
    void step(double dt);
    bool isDone() const;

    struct RewardInfo {
        double height;
        double distance_from_origin;
        double velocity_magnitude;
        double angle_magnitude;
        double angular_velocity_magnitude;
        bool crashed;
        bool out_of_bounds;
    };
    RewardInfo getRewardInfo() const;
    void printState() const;

private:
    double motorThrust(int i) const;
    void updateMotors(double dt);
    void normalizeAngles();
};