import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. DEFINIR TRAJETÓRIA REAL
# --------------------------------------------------

trajetoria_real = np.array([
    [0, 0],
    [1, 0],
    [2, 0],
    [2, 1],
    [2, 2],
    [1, 2],
    [0, 2],
    [0, 1],
    [0, 0]
])

N = len(trajetoria_real)

# --------------------------------------------------
# 2. GERAR ODOMETRIA COM RUÍDO
# --------------------------------------------------

np.random.seed(42)

trajetoria_ruidosa = []

# adicionar primeira pose
trajetoria_ruidosa.append(trajetoria_real[0])

for i in range(1, N):

    # deslocamento real
    dx = trajetoria_real[i][0] - trajetoria_real[i - 1][0]
    dy = trajetoria_real[i][1] - trajetoria_real[i - 1][1]

    # adicionar ruído
    dx = dx + np.random.normal(0, 0.1)
    dy = dy + np.random.normal(0, 0.1)

    # nova pose
    nova_pose = trajetoria_ruidosa[-1] + np.array([dx, dy])

    trajetoria_ruidosa.append(nova_pose)

trajetoria_ruidosa = np.array(trajetoria_ruidosa)

# --------------------------------------------------
# 3. CRIAR MATRIZ Ω E VETOR ξ
# --------------------------------------------------

Omega = np.zeros((2 * N, 2 * N))
Xi = np.zeros((2 * N, 1))

# --------------------------------------------------
# 4. FIXAR PRIMEIRA POSE
# --------------------------------------------------

Omega[0, 0] = 1
Omega[1, 1] = 1

Xi[0] = trajetoria_ruidosa[0][0]
Xi[1] = trajetoria_ruidosa[0][1]

# --------------------------------------------------
# 5. ADICIONAR RESTRIÇÕES DE ODOMETRIA
# --------------------------------------------------

for i in range(N - 1):

    dx = trajetoria_ruidosa[i + 1][0] - trajetoria_ruidosa[i][0]
    dy = trajetoria_ruidosa[i + 1][1] - trajetoria_ruidosa[i][1]

    ix = 2 * i
    iy = 2 * i + 1

    jx = 2 * (i + 1)
    jy = 2 * (i + 1) + 1

    # atualizar matriz Ω
    Omega[ix, ix] += 1
    Omega[iy, iy] += 1

    Omega[jx, jx] += 1
    Omega[jy, jy] += 1

    Omega[ix, jx] -= 1
    Omega[iy, jy] -= 1

    Omega[jx, ix] -= 1
    Omega[jy, iy] -= 1

    # atualizar vetor ξ
    Xi[ix] -= dx
    Xi[iy] -= dy

    Xi[jx] += dx
    Xi[jy] += dy

# --------------------------------------------------
# 6. ADICIONAR LOOP CLOSURE
# --------------------------------------------------

peso_loop = 5

ix = 0
iy = 1

jx = 2 * (N - 1)
jy = 2 * (N - 1) + 1

dx = 0
dy = 0

Omega[ix, ix] += peso_loop
Omega[iy, iy] += peso_loop

Omega[jx, jx] += peso_loop
Omega[jy, jy] += peso_loop

Omega[ix, jx] -= peso_loop
Omega[iy, jy] -= peso_loop

Omega[jx, ix] -= peso_loop
Omega[jy, iy] -= peso_loop

Xi[ix] -= peso_loop * dx
Xi[iy] -= peso_loop * dy

Xi[jx] += peso_loop * dx
Xi[jy] += peso_loop * dy

# --------------------------------------------------
# 7. RESOLVER SISTEMA
# --------------------------------------------------

mu = np.linalg.inv(Omega) @ Xi

# --------------------------------------------------
# 8. EXTRAIR POSES OTIMIZADAS
# --------------------------------------------------

trajetoria_otimizada = []

for i in range(N):

    x = mu[2 * i][0]
    y = mu[2 * i + 1][0]

    trajetoria_otimizada.append([x, y])

trajetoria_otimizada = np.array(trajetoria_otimizada)

# --------------------------------------------------
# 9. MOSTRAR RESULTADOS
# --------------------------------------------------

plt.figure(figsize=(8, 8))

# trajetória real
plt.plot(
    trajetoria_real[:, 0],
    trajetoria_real[:, 1],
    'g-o',
    label='Trajetória Real'
)

# trajetória ruidosa
plt.plot(
    trajetoria_ruidosa[:, 0],
    trajetoria_ruidosa[:, 1],
    'r--o',
    label='Odometriа com Ruído'
)

# trajetória otimizada
plt.plot(
    trajetoria_otimizada[:, 0],
    trajetoria_otimizada[:, 1],
    'b-o',
    linewidth=3,
    label='Trajetória Otimizada'
)

# conexões do grafo
for i in range(N - 1):

    plt.plot(
        [trajetoria_otimizada[i][0], trajetoria_otimizada[i + 1][0]],
        [trajetoria_otimizada[i][1], trajetoria_otimizada[i + 1][1]],
        'k:',
        alpha=0.5
    )

# loop closure
plt.plot(
    [trajetoria_otimizada[-1][0], trajetoria_otimizada[0][0]],
    [trajetoria_otimizada[-1][1], trajetoria_otimizada[0][1]],
    'm--',
    linewidth=2,
    label='Loop Closure'
)

plt.xlim(-1, 3)
plt.ylim(-1, 3)

plt.grid(True)
plt.axis('equal')

plt.title('Pose Graph SLAM 2D')
plt.legend()

plt.show()