import numpy as np
import matplotlib.pyplot as plt

# Drone parameters
m, I, g, dt = 1.0, 0.02, 9.81, 0.002     # mass, inertia, gravity, time step
target_x, target_z = 5.0, 5.0            # where we want to go (m)

# Altitude PID (same idea as before)
kp_z, ki_z, kd_z = 8.0, 2.0, 5.0
# Position loop (x error -> desired tilt)
kp_x, kd_x = 0.1, 0.2
# Attitude loop (tilt error -> torque), must be faster than the position loop
kp_t, kd_t = 0.05, 0.15
max_tilt = 0.4                           # rad (~23 deg)

x = z = th = 0.0                         # position, height, pitch angle
vx = vz = w = 0.0                        # velocities, angular rate
int_z, prev_ez = 0.0, target_z

log = {"t": [], "x": [], "z": [], "th": [], "th_des": []}

for step in range(int(15 / dt)):
    # --- Outer loop: horizontal position -> desired pitch ---
    th_des = kp_x * (target_x - x) - kd_x * vx
    th_des = np.clip(th_des, -max_tilt, max_tilt)

    # --- Altitude PID -> thrust (boosted to compensate for tilt) ---
    ez = target_z - z
    dz = (ez - prev_ez) / dt
    prev_ez = ez
    T_raw = m * g + kp_z * ez + ki_z * int_z + kd_z * dz
    T = np.clip(T_raw / max(np.cos(th), 0.5), 0, 2.5 * m * g)
    if 0 < T < 2.5 * m * g:              # anti-windup
        int_z += ez * dt

    # --- Inner loop: pitch error -> torque ---
    tau = kp_t * (th_des - th) - kd_t * w

    # --- Physics ---
    ax = T * np.sin(th) / m
    az = T * np.cos(th) / m - g
    alpha = tau / I

    vx += ax * dt;  vz += az * dt;  w += alpha * dt
    x += vx * dt;   z += vz * dt;   th += w * dt
    if z <= 0 and vz < 0:                # ground
        z, vz = 0.0, 0.0

    if step % 10 == 0:
        log["t"].append(step * dt); log["x"].append(x); log["z"].append(z)
        log["th"].append(np.degrees(th)); log["th_des"].append(np.degrees(th_des))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(log["x"], log["z"]); a1.plot(target_x, target_z, "rx", ms=12)
a1.set_xlabel("x (m)"); a1.set_ylabel("height (m)"); a1.set_title("Flight path")
a1.grid(True)
a2.plot(log["t"], log["th"], label="actual pitch")
a2.plot(log["t"], log["th_des"], "--", label="desired pitch")
a2.set_xlabel("time (s)"); a2.set_ylabel("pitch (deg)"); a2.legend()
a2.set_title("Attitude tracking"); a2.grid(True)
plt.tight_layout(); plt.show()