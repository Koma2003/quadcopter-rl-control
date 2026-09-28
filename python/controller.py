import numpy as np


class QuadController:
    def __init__(
        self,
        kp_roll=6.0,
        kd_roll=2.0,
        kp_pitch=6.0,
        kd_pitch=2.0,
        kp_yaw=2.0,
        kd_yaw=0.5,

        max_roll_cmd=0.4,
        max_pitch_cmd=0.4,

        throttle_base=0.4953,
        throttle_scale=0.15,

        kvz=0.025,

        motor_min=0.0,
        motor_max=1.0,
    ):
        self.kp_roll = kp_roll
        self.kd_roll = kd_roll
        self.kp_pitch = kp_pitch
        self.kd_pitch = kd_pitch
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw

        self.max_roll_cmd = max_roll_cmd
        self.max_pitch_cmd = max_pitch_cmd

        self.throttle_base = throttle_base
        self.throttle_scale = throttle_scale
        self.kvz = kvz

        self.motor_min = motor_min
        self.motor_max = motor_max

        self.target_yaw = 0.0

    def reset(self, initial_yaw=0.0):
        self.target_yaw = initial_yaw

    def compute(self, action, drone):
        action = np.clip(action, -1.0, 1.0)

        throttle_delta = float(action[0])
        roll_cmd = float(action[1]) * self.max_roll_cmd
        pitch_cmd = float(action[2]) * self.max_pitch_cmd

        roll_err = roll_cmd - drone.roll
        u_roll = self.kp_roll * roll_err - self.kd_roll * drone.roll_rate

        pitch_err = pitch_cmd - drone.pitch
        u_pitch = self.kp_pitch * pitch_err - self.kd_pitch * drone.pitch_rate

        yaw_err = self._wrap_angle(self.target_yaw - drone.yaw)
        u_yaw = self.kp_yaw * yaw_err - self.kd_yaw * drone.yaw_rate

        throttle = (
            self.throttle_base
            + self.throttle_scale * throttle_delta
            - self.kvz * drone.vz
        )

        mix_scale = 0.25

        m0 = throttle - mix_scale * u_pitch + mix_scale * u_yaw
        m1 = throttle + mix_scale * u_roll  - mix_scale * u_yaw
        m2 = throttle + mix_scale * u_pitch + mix_scale * u_yaw
        m3 = throttle - mix_scale * u_roll  - mix_scale * u_yaw

        motors = [
            np.clip(m0, self.motor_min, self.motor_max),
            np.clip(m1, self.motor_min, self.motor_max),
            np.clip(m2, self.motor_min, self.motor_max),
            np.clip(m3, self.motor_min, self.motor_max),
        ]
        return motors

    @staticmethod
    def _wrap_angle(a):
        while a > np.pi:
            a -= 2.0 * np.pi
        while a < -np.pi:
            a += 2.0 * np.pi
        return a