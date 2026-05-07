import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. DEFINIR FUNÇÃO DE ROTAÇÃO
# --------------------------------------------------

def MATRIZ_ROTACAO(angulo_em_graus):
    theta = np.deg2rad(angulo_em_graus)

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    return R

# --------------------------------------------------
# 2. CRIAR NUVEM DE REFERÊNCIA A
# --------------------------------------------------

A = np.array([
    [0,0], [1,0], [2,0], [3,0],
    [3,1], [3,2],
    [2,3], [1,3], [0,3],
    [-0.5,2], [-1,1],
    [1.2,1.4], [2.1,1.7]
])

# --------------------------------------------------
# 3. GERAR NUVEM TRANSFORMADA B
# --------------------------------------------------

rot_real = 8
trans_real = np.array([0.8, 0.6])

R_real = MATRIZ_ROTACAO(rot_real)

B = []

for ponto in A:
    ponto_rot = R_real @ ponto
    ponto_B = ponto_rot + trans_real
    B.append(ponto_B)

B = np.array(B)

# erro inicial
B = B + np.array([-0.3, 0.2])

# --------------------------------------------------
# 4. EXIBIR NUVENS INICIAIS
# --------------------------------------------------

plt.figure()
plt.scatter(A[:,0], A[:,1], label='A')
plt.scatter(B[:,0], B[:,1], label='B')
plt.title("Inicial")
plt.legend()
plt.xlim(-5, 25)
plt.ylim(-5, 25)
plt.gca().set_aspect('equal')
plt.show()

# --------------------------------------------------
# 5. CONFIGURAR ICP
# --------------------------------------------------

max_iter = 20
tolerancia = 0.00001

# --------------------------------------------------
# 6. LOOP PRINCIPAL DO ICP
# --------------------------------------------------

plt.ion()
fig, ax = plt.subplots(figsize=(5,5))

for it in range(max_iter):

    # --------------------------------------------------
    # 6.1 ENCONTRAR CORRESPONDÊNCIAS
    # --------------------------------------------------

    matched_A = []

    for pb in B:

        menor_dist = float('inf')
        melhor_ponto = None

        for pa in A:
            dist = np.linalg.norm(pb - pa)

            if dist < menor_dist:
                menor_dist = dist
                melhor_ponto = pa

        matched_A.append(melhor_ponto)

    matched_A = np.array(matched_A)

    # --------------------------------------------------
    # 6.2 CALCULAR CENTROIDES
    # --------------------------------------------------

    centroide_A = np.mean(matched_A, axis=0)
    centroide_B = np.mean(B, axis=0)

    # --------------------------------------------------
    # 6.3 CENTRALIZAR PONTOS
    # --------------------------------------------------

    AA = matched_A - centroide_A
    BB = B - centroide_B

    # --------------------------------------------------
    # 6.4 MATRIZ DE COVARIÂNCIA
    # --------------------------------------------------

    H = BB.T @ AA

    # --------------------------------------------------
    # 6.5 CALCULAR ROTAÇÃO ÓTIMA (SVD)
    # --------------------------------------------------

    U, S, VT = np.linalg.svd(H)

    R = VT.T @ U.T

    if np.linalg.det(R) < 0:
        VT[1, :] *= -1
        R = VT.T @ U.T

    # --------------------------------------------------
    # 6.6 CALCULAR TRANSLAÇÃO
    # --------------------------------------------------

    t = centroide_A - R @ centroide_B

    # --------------------------------------------------
    # 6.7 ATUALIZAR NUVEM B
    # --------------------------------------------------

    novo_B = []

    for ponto in B:
        novo_ponto = R @ ponto + t
        novo_B.append(novo_ponto)

    B = np.array(novo_B)

    # --------------------------------------------------
    # 6.8 CALCULAR ERRO MÉDIO
    # --------------------------------------------------

    erro = np.mean(np.linalg.norm(B - matched_A, axis=1))

    print(f"Iteração {it+1} | erro = {erro:.6f}")

    # --------------------------------------------------
    # 6.9 MOSTRAR RESULTADO PARCIAL
    # --------------------------------------------------

    ax.clear()
    ax.scatter(A[:,0], A[:,1], label='A')
    ax.scatter(B[:,0], B[:,1], label='B alinhado')
    ax.set_title(f"Iteração {it+1}")
    ax.legend()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    plt.pause(1)

    # --------------------------------------------------
    # 6.10 TESTE DE PARADA
    # --------------------------------------------------

    if erro < tolerancia:
        break

# --------------------------------------------------
# 7. FIM
# --------------------------------------------------
plt.ioff()
plt.show()
print("Alinhamento concluído.")