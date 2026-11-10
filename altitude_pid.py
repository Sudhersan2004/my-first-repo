import matplotlib.pyplot as plt

m, g, dt = 1.0, 9.81, 0.01        # mass (kg), gravity, time step (s)
kp, ki, kd = 30.0, 0.0, 0.0        # PID gains
target = 10.0                     # desired altitude (m)

z, vz = 0.0, 0.0                  # altitude and vertical speed
integral, prev_err = 0.0, target - z
t_log, z_log, thrust_log = [], [], []

for step in range(1500):          # 15 seconds
    err = target - z
    deriv = (err - prev_err) / dt
    prev_err = err

    # Thrust = hover thrust (m*g) + PID correction
    thrust = m * g + kp * err + ki * integral + kd * deriv

    # Motors can't push infinitely hard: limit thrust to 0..2x weight
    limited = max(0.0, min(thrust, 2 * m * g))

    # Anti-windup: only accumulate the integral when not saturated
    if limited == thrust:
        integral += err * dt

    az = limited / m - g          # Newton: F = ma
    vz += az * dt
    z += vz * dt
    z = max(z, 0.0)               # the ground stops the drone

    t_log.append(step * dt)
    z_log.append(z)
    thrust_log.append(limited)

fig, (a1, a2) = plt.subplots(2, 1, sharex=True)
a1.plot(t_log, z_log); a1.axhline(target, ls="--", c="r")
a1.set_ylabel("Altitude (m)")
a2.plot(t_log, thrust_log); a2.set_ylabel("Thrust (N)")
a2.set_xlabel("Time (s)")
plt.show()