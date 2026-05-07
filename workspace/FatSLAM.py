import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURAÇÕES
# -------------------------------

N = 42

landmarks_real = np.array([
    [5, 8],
    [10, 4],
    [12, 10],
    [16, 19],
    [5, 5],
    [20, 12],
    [22, 22],
    [22, 2]
])

motion_noise = [0.1, np.deg2rad(2)]
sensor_noise = [0.3, np.deg2rad(5)]

# -------------------------------
# PARTÍCULAS
# -------------------------------

particles = []

for _ in range(N):
    particles.append({
        "x": np.random.normal(0, 0.1),
        "y": np.random.normal(0, 0.1),
        "theta": np.random.normal(0, 0.05),
        "w": 1.0 / N,
        "landmarks": [None] * len(landmarks_real),
        "cov": [None] * len(landmarks_real)
    })

# -------------------------------
# ROBÔ REAL
# -------------------------------

x_real, y_real, theta_real = 0, 0, 0

# -------------------------------
# MOVIMENTOS
# -------------------------------

base_motion = [
    (1, np.deg2rad(10)),
    (1, np.deg2rad(0)),
    (1, np.deg2rad(-5)),
    (1, np.deg2rad(0)),
    (1, np.deg2rad(8)),
    (1, np.deg2rad(0)),
]

motions = base_motion * 5

# -------------------------------
# ATIVAR ANIMAÇÃO
# -------------------------------

plt.ion()
plt.figure()

# -------------------------------
# LOOP PRINCIPAL
# -------------------------------

for step, (v, w) in enumerate(motions):

    # 1) ROBÔ REAL
    theta_real += w
    x_real += v * np.cos(theta_real)
    y_real += v * np.sin(theta_real)

    # 2) PARTÍCULAS
    for p in particles:

        v_n = v + np.random.normal(0, motion_noise[0])
        w_n = w + np.random.normal(0, motion_noise[1])

        p["theta"] += w_n
        p["x"] += v_n * np.cos(p["theta"])
        p["y"] += v_n * np.sin(p["theta"])

    # 3) SENSOR
    measurements = []

    for lx, ly in landmarks_real:
        dx = lx - x_real
        dy = ly - y_real

        r = np.sqrt(dx**2 + dy**2)
        b = np.arctan2(dy, dx) - theta_real

        r += np.random.normal(0, sensor_noise[0])
        b += np.random.normal(0, sensor_noise[1])

        measurements.append((r, b))

    # 4) UPDATE
    for p in particles:

        p["w"] = 1.0

        for j, (r, b) in enumerate(measurements):

            if p["landmarks"][j] is None:

                mx = p["x"] + r * np.cos(p["theta"] + b)
                my = p["y"] + r * np.sin(p["theta"] + b)

                p["landmarks"][j] = np.array([mx, my])
                p["cov"][j] = np.eye(2)

            else:

                mu = p["landmarks"][j]
                Sigma = p["cov"][j]

                dx = mu[0] - p["x"]
                dy = mu[1] - p["y"]

                q = dx**2 + dy**2
                dist = np.sqrt(q)

                z_hat = np.array([
                    dist,
                    np.arctan2(dy, dx) - p["theta"]
                ])

                H = np.array([
                    [dx/dist, dy/dist],
                    [-dy/q, dx/q]
                ])

                R = np.diag(sensor_noise)

                S = H @ Sigma @ H.T + R
                K = Sigma @ H.T @ np.linalg.inv(S)

                z = np.array([r, b])
                error = z - z_hat

                mu = mu + K @ error
                Sigma = (np.eye(2) - K @ H) @ Sigma

                p["landmarks"][j] = mu
                p["cov"][j] = Sigma

                p["w"] *= np.exp(-0.5 * error.T @ np.linalg.inv(S) @ error)

    # 5) NORMALIZAR
    total = sum(p["w"] for p in particles)
    for p in particles:
        p["w"] /= total

    # 6) REAMOSTRAGEM
    weights = [p["w"] for p in particles]
    indices = np.random.choice(range(N), N, p=weights)

    particles = [particles[i].copy() for i in indices]

    # -------------------------------
    # PLOT EM TEMPO REAL (FIXO)
    # -------------------------------

    plt.clf()

    ax = plt.gca()
    ax.set_xlim(-5, 25)
    ax.set_ylim(-5, 25)
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)

    # partículas
    for p in particles:
        ax.scatter(p["x"], p["y"], c='blue', s=10)

    # robô
    ax.scatter(x_real, y_real, c='red', label="Robô")

    # landmarks reais
    ax.scatter(landmarks_real[:,0], landmarks_real[:,1],
            c='green', marker='x', label="Reais")

    # landmarks estimados
    for j in range(len(landmarks_real)):
        xs, ys = [], []

        for p in particles:
            if p["landmarks"][j] is not None:
                xs.append(p["landmarks"][j][0])
                ys.append(p["landmarks"][j][1])

        if xs:
            ax.scatter(np.mean(xs), np.mean(ys),
                    c='purple', label="Estimado" if j == 0 else "")

    # título com passo
    ax.set_title(f"FastSLAM - Passo {step+1} / {len(motions)}")

    ax.legend()
    plt.pause(0.3)
# manter janela aberta
plt.ioff()
plt.show()