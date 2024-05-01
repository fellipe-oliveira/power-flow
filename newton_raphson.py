import pandas as pd
import numpy as np

def newton_raphson(rede, tol: float, num_iter_max: int) -> None:
    iteracao_nova = pd.concat(
        [rede.fase.loc[rede.barras_PQ_ou_PV.index], rede.V.loc[rede.barras_PQ.index]])
    erro = np.inf
    num_iter = 0
    while(erro > tol and num_iter < num_iter_max):
        iteracao = iteracao_nova.copy()

        J_fP_fase = np.array([[dfP_dfase(rede,
            i, k) for k in rede.barras_PQ_ou_PV.index] for i in rede.barras_PQ_ou_PV.index])
        J_fP_V = np.array([[dfP_dV(rede, i, k) for k in rede.barras_PQ.index]
                           for i in rede.barras_PQ_ou_PV.index])
        J_fQ_fase = np.array([[dfQ_dfase(rede,
            i, k) for k in rede.barras_PQ_ou_PV.index] for i in rede.barras_PQ.index])
        J_fQ_V = np.array(
            [[dfQ_dV(rede, i, k) for k in rede.barras_PQ.index] for i in rede.barras_PQ.index])

        J = np.concatenate(
            (
                np.concatenate((J_fP_fase, J_fP_V), axis=1),
                np.concatenate((J_fQ_fase, J_fQ_V), axis=1)
            )
        )

        Jinv = pd.DataFrame(np.linalg.inv(J), index=rede.barras_PQ_ou_PV.index.tolist()+rede.barras_PQ.index.tolist(),
                            columns=rede.barras_PQ_ou_PV.index.tolist()+rede.barras_PQ.index.tolist())

        rede.cargas['V'] = rede.V.loc[rede.cargas['Barra']].values
        rede.cargas['P_atual'] = rede.cargas['P']*(
            rede.cargas['cP']*rede.cargas['V']**2+rede.cargas['bP']*rede.cargas['V']+rede.cargas['aP'])
        rede.cargas['Q_atual'] = rede.cargas['Q']*(
            rede.cargas['cQ']*rede.cargas['V']**2+rede.cargas['bQ']*rede.cargas['V']+rede.cargas['aQ'])
        rede.PL = pd.Series([sum(rede.cargas.loc[rede.cargas['Barra'] == barra]['P'])
                             for barra in rede.barras.index], index=rede.barras.index)
        rede.QL = pd.Series([sum(rede.cargas.loc[rede.cargas['Barra'] == barra]['Q'])
                             for barra in rede.barras.index], index=rede.barras.index)

        fP_fQ = pd.Series([fP(rede, barra) for barra in rede.barras_PQ_ou_PV.index]+[fQ(rede, barra) for barra in rede.barras_PQ.index],
                          index=rede.barras_PQ_ou_PV.index.tolist()+rede.barras_PQ.index.tolist())
        iteracao_nova = iteracao - Jinv @ fP_fQ
        rede.fase.loc[rede.barras_PQ_ou_PV.index] = iteracao_nova.iloc[:len(
            rede.barras_PQ_ou_PV.index)].loc[rede.barras_PQ_ou_PV.index]
        rede.V.loc[rede.barras_PQ.index] = iteracao_nova.iloc[len(
            rede.barras_PQ_ou_PV.index):].loc[rede.barras_PQ.index]
        num_iter += 1
        erro = max(abs(iteracao_nova-iteracao))

    rede.cargas['fase'] = rede.fase.loc[rede.cargas['Barra']].values
    print(f'\nNúmero de iterações: {num_iter}')
    if (num_iter == num_iter_max):
        print('Solução divergente.')
    else:
        print('Solução convergente.')
        
# Derivadas parciais do jacobiano

def dfP_dfase(rede, i: int, k: int):
    if i != k:
        return rede.V[i]*rede.V[k]*rede.Ybarra.mod.at[i, k]*np.sin(rede.fase[i]-rede.fase[k]-rede.Ybarra.ang.at[i, k])
    else:
        return -sum([rede.V[i]*rede.V[j]*rede.Ybarra.mod.at[i, j]*np.sin(rede.fase[i]-rede.fase[j]-rede.Ybarra.ang.at[i, j]) for j in rede.barras.index if j != i])

def dfP_dV(rede, i: int, k: int):
    if i != k:
        return rede.V[i]*rede.Ybarra.mod.at[i, k]*np.cos(rede.fase[i]-rede.fase[k]-rede.Ybarra.ang.at[i, k])
    else:
        return 2*rede.V[i]*rede.Ybarra.mod.at[i, i]*np.cos(rede.Ybarra.ang.at[i, i])+sum([rede.V[j]*rede.Ybarra.mod.at[i, j]*np.cos(rede.fase[i]-rede.fase[j]-rede.Ybarra.ang.at[i, j]) for j in rede.barras.index if j != i])

def dfQ_dfase(rede, i: int, k: int):
    if i != k:
        return -rede.V[i]*rede.V[k]*rede.Ybarra.mod.at[i, k]*np.cos(rede.fase[i]-rede.fase[k]-rede.Ybarra.ang.at[i, k])
    else:
        return sum([rede.V[i]*rede.V[j]*rede.Ybarra.mod.at[i, j]*np.cos(rede.fase[i]-rede.fase[j]-rede.Ybarra.ang.at[i, j]) for j in rede.barras.index if j != i])

def dfQ_dV(rede, i: int, k: int):
    if i != k:
        return rede.V[i]*rede.Ybarra.mod.at[i, k]*np.sin(rede.fase[i]-rede.fase[k]-rede.Ybarra.ang.at[i, k])
    else:
        return -2*rede.V[i]*rede.Ybarra.mod.at[i, i]*np.sin(rede.Ybarra.ang.at[i, i])+sum([rede.V[j]*rede.Ybarra.mod.at[i, j]*np.sin(rede.fase[i]-rede.fase[j]-rede.Ybarra.ang.at[i, j]) for j in rede.barras.index if j != i])

# Funções para as quais se procuram as raízes no método de Newton-Raphson

def fP(rede, i: int):
    return sum([rede.V[i]*rede.V[j]*rede.Ybarra.mod.at[i, j]*np.cos(rede.fase[i]-rede.fase[j]-rede.Ybarra.ang.at[i, j]) for j in rede.barras.index])+rede.PL[i]-rede.PG[i]

def fQ(rede, i: int):
    return sum([rede.V[i]*rede.V[j]*rede.Ybarra.mod.at[i, j]*np.sin(rede.fase[i]-rede.fase[j]-rede.Ybarra.ang.at[i, j]) for j in rede.barras.index])+rede.QL[i]-rede.QG[i]