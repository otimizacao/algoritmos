import random
import numpy as np
import math
import matplotlib.pyplot as plt

# ==========================================================
# 1. GERAÇÃO DOS ATIVOS DE SEGURANÇA
# ==========================================================
ativos_seguranca = {}

for i in range(200):
    ativos_seguranca[f"Servidor_{i+1}"] = random.randint(50, 300)

ativos_seguranca["Database"] = 200
ativos_seguranca["Backup_Node"] = 180

valores = list(ativos_seguranca.values())
nomes = list(ativos_seguranca.keys())
n_elementos = len(valores)
carga_total = sum(valores)
LIMITE_FIREWALL = int((carga_total / 2) * 1.10)

print(f"Carga Total: {carga_total}")
print(f"Capacidade Firewall: {LIMITE_FIREWALL}")

# ==========================================================
# 2. FUNÇÃO OBJETIVO COM RESTRIÇÕES
# ==========================================================
def calcular_energia_com_restricoes(solucao):
    carga_fw0 = sum(valores[i] for i in range(n_elementos) if solucao[i] == 0)
    carga_fw1 = sum(valores[i] for i in range(n_elementos) if solucao[i] == 1)

    # Objetivo principal: equilibrar carga
    energia = abs(carga_fw0 - carga_fw1)

    # Restrição 1: Failover
    if carga_fw0 > LIMITE_FIREWALL:
        energia += (carga_fw0 - LIMITE_FIREWALL) * 5
    if carga_fw1 > LIMITE_FIREWALL:
        energia += (carga_fw1 - LIMITE_FIREWALL) * 5

    # Restrição 2: Antiafinidade
    idx_db = nomes.index("Database")
    idx_bkp = nomes.index("Backup_Node")
    if solucao[idx_db] == solucao[idx_bkp]:
        energia += 300

    return energia

# ==========================================================
# 3. ALGORITMO 1: SIMULATED ANNEALING MELHORADO
# ==========================================================
def simulated_annealing(T_inicial, T_final, alfa, max_iter, solucao_inicial):
    solucao_atual = solucao_inicial.copy()
    energia_atual = calcular_energia_com_restricoes(solucao_atual)

    melhor_solucao = solucao_atual.copy()
    melhor_energia = energia_atual

    T = T_inicial
    historico_energia = []
    iteracoes_totais = 0

    while T > T_final:
        for _ in range(max_iter):
            iteracoes_totais += 1
            vizinho = solucao_atual.copy()
            idx = np.random.randint(n_elementos)

            # Bit-flip
            vizinho[idx] = 1 - vizinho[idx]
            energia_vizinho = calcular_energia_com_restricoes(vizinho)
            diff_energia = energia_vizinho - energia_atual

            # Critério de Metropolis
            if diff_energia < 0 or np.random.rand() < math.exp(-diff_energia / T):
                solucao_atual = vizinho
                energia_atual = energia_vizinho

                if energia_atual < melhor_energia:
                    melhor_energia = energia_atual
                    melhor_solucao = solucao_atual.copy()

            historico_energia.append(energia_atual)
        T *= alfa

    return melhor_solucao, melhor_energia, historico_energia, iteracoes_totais

# ==========================================================
# 4. ALGORITMO 2: HILL CLIMBING (BUSCA LOCAL)
# ==========================================================
def hill_climbing(max_iter_totais, solucao_inicial):
    solucao_atual = solucao_inicial.copy()
    energia_atual = calcular_energia_com_restricoes(solucao_atual)

    historico_energia = []

    for _ in range(max_iter_totais):
        vizinho = solucao_atual.copy()
        idx = np.random.randint(n_elementos)

        # Bit-flip
        vizinho[idx] = 1 - vizinho[idx]
        energia_vizinho = calcular_energia_com_restricoes(vizinho)

        # Aceita apenas se for estritamente melhor (ou igual)
        if energia_vizinho <= energia_atual:
            solucao_atual = vizinho
            energia_atual = energia_vizinho

        historico_energia.append(energia_atual)

    return solucao_atual, energia_atual, historico_energia

# ==========================================================
# 5. EXECUÇÃO COMPARATIVA
# ==========================================================
# Definimos a mesma solução inicial para garantir uma comparação justa
solucao_inicial_compartilhada = np.random.randint(2, size=n_elementos)

# Executa Simulated Annealing
solucao_sa, energia_sa, historico_sa, total_iter_sa = simulated_annealing(
    T_inicial=100, T_final=0.01, alfa=0.98, max_iter=30, solucao_inicial=solucao_inicial_compartilhada
)

# Executa Hill Climbing com o mesmo número de iterações do SA
solucao_hc, energia_hc, historico_hc = hill_climbing(
    max_iter_totais=total_iter_sa, solucao_inicial=solucao_inicial_compartilhada
)

# ==========================================================
# 6. EXIBIÇÃO DOS RESULTADOS COMPARATIVOS
# ==========================================================
def extrair_metricas(solucao):
    fw0 = sum(valores[i] for i in range(n_elementos) if solucao[i] == 0)
    fw1 = sum(valores[i] for i in range(n_elementos) if solucao[i] == 1)
    return fw0, fw1

fw0_sa, fw1_sa = extrair_metricas(solucao_sa)
fw0_hc, fw1_hc = extrair_metricas(solucao_hc)

print("\n=======================================================")
print("             COMPARAÇÃO DOS ALGORITMOS                 ")
print("=======================================================")
print(f"Métrica                           | SA          | Hill Climbing")
print(f"----------------------------------|-------------|--------------")
print(f"Carga no FW0 (Mbps)               | {fw0_sa:<11} | {fw0_hc}")
print(f"Carga no FW1 (Mbps)               | {fw1_sa:<11} | {fw1_hc}")
print(f"Diferença Absoluta (Mbps)         | {abs(fw0_sa - fw1_sa):<11} | {abs(fw0_hc - fw1_hc)}")
print(f"Energia Final ( Penalidades )     | {energia_sa:<11} | {energia_hc}")
print(f"Total de Iterações Executadas     | {total_iter_sa:<11} | {total_iter_sa}")
print("=======================================================")

# ==========================================================
# 7. CURVA DE CONVERGÊNCIA COMPARATIVA
# ==========================================================
plt.figure(figsize=(12, 6))
plt.plot(historico_sa, label=f"Simulated Annealing (Final: {energia_sa})", color="blue", alpha=0.8)
plt.plot(historico_hc, label=f"Hill Climbing (Final: {energia_hc})", color="orange", alpha=0.8)
plt.title("Comparação de Convergência: Simulated Annealing vs Hill Climbing")
plt.xlabel("Iterações")
plt.ylabel("Energia (Menor é Melhor)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.show()
