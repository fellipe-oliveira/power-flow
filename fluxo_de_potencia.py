import pandas as pd
import numpy as np
from rede import Rede
import newton_raphson as nr
from sys import argv

########
# Este fluxo de potência tem o seguinte requisito: apenas
# um gerador equivalente por barra é permitido.
########

def polar_to_complex(r, theta):
    return r*np.cos(theta) + 1j*r*np.sin(theta)

# Entrada de dados

entrada = pd.read_excel(argv[1], None)

# Criação de um objeto do tipo Rede para a entrada selecionada

rede = Rede(entrada, sb=1, f=60)

# Cálculo das tensões na rede

nr.newton_raphson(rede, tol=1E-5,num_iter_max=10)

Vformatted = [f'{round(rede.V[k], 4)} /_ {round(rede.fase[k]*180/np.pi, 4)}° pu' for k in rede.barras.index]
rede.barras['Tensão calculada'] = Vformatted

# Cálculo das correntes nos equipamentos

for idx in rede.linhas.index:
    de, para, Ys, Ysh = (rede.linhas.loc[idx, 'De'], 
                        rede.linhas.loc[idx, 'Para'], 
                        rede.linhas.loc[idx, 'Y'],
                        rede.linhas.loc[idx, 'Ysh'])
    
    Y_equip = pd.DataFrame(0, columns=[de, para], index=[de, para], dtype=complex)
    V_equip = pd.Series(0, index=[de, para], dtype=complex)
    
    Y_equip.at[de, de] += Ys + Ysh/2
    Y_equip.at[de, para] += -Ys
    Y_equip.at[para, de] += -Ys
    Y_equip.at[para, para] += Ys + Ysh/2
    
    V_equip[de] = polar_to_complex(rede.V[de], rede.fase[de])
    V_equip[para] = polar_to_complex(rede.V[para], rede.fase[para])
    
    I_equip = Y_equip @ V_equip
    
    Vn = rede.linhas.loc[idx, 'Vn (kV)']
    
    rede.linhas.loc[idx, 'V (de)'] = f'{round(rede.V[de], 4)} /_ {round(rede.fase[de]*180/np.pi, 3)}° pu'
    rede.linhas.loc[idx, 'V (para)'] = f'{round(rede.V[para], 4)} /_ {round(rede.fase[para]*180/np.pi, 3)}° pu'
    rede.linhas.loc[idx, 'I (de)'] = f'{round(abs(I_equip[de])*1000*rede.sb/(Vn*3**0.5), 2)} /_ {round(np.angle(I_equip[de])*180/np.pi, 2)}° A'
    rede.linhas.loc[idx, 'I (para)'] = f'{round(abs(I_equip[para])*1000*rede.sb/(Vn*3**0.5), 2)} /_ {round(np.angle(I_equip[para])*180/np.pi, 2)}° A'

for idx in rede.transformadores.index:
    de, para, Ys, alfa, beta = (rede.transformadores.loc[idx, 'De'], rede.transformadores.loc[idx, 'Para'],
                                rede.transformadores.loc[idx, 'Y'], rede.transformadores.loc[idx, 'alfa'], 
                                rede.transformadores.loc[idx, 'beta'])
    
    Y_equip = pd.DataFrame(0, columns=[de, para], index=[de, para], dtype=complex)
    V_equip = pd.Series(0, index=[de, para], dtype=complex)
    
    Y_equip.at[de, de] += Ys
    Y_equip.at[de, para] += -Ys*alfa/beta
    Y_equip.at[para, de] += -Ys*np.conj(alfa/beta)
    Y_equip.at[para, para] += Ys*abs(alfa/beta)**2
    
    V_equip[de] = polar_to_complex(rede.V[de], rede.fase[de])
    V_equip[para] = polar_to_complex(rede.V[para], rede.fase[para])
    
    I_equip = Y_equip @ V_equip
    
    sn_tf = rede.transformadores.loc[idx, 'Sn (MVA)']
    rede.transformadores.loc[idx, 'V (de)'] = f'{round(rede.V[de], 4)} /_ {round(rede.fase[de]*180/np.pi, 4)}° pu'
    rede.transformadores.loc[idx, 'V (para)'] = f'{round(rede.V[para], 4)} /_ {round(rede.fase[para]*180/np.pi, 4)}° pu'
    rede.transformadores.loc[idx, 'I'] = f'{round(abs(I_equip[de]*rede.sb/sn_tf), 4)} /_ {round(np.angle(I_equip[de])*180/np.pi, 2)}° pu'

for idx in rede.reatores.index:
    Yr = rede.reatores.loc[idx, 'Y']
    barra = rede.reatores.loc[idx, 'Barra']
    Vr = polar_to_complex(rede.V[barra], rede.fase[barra])
    
    I = Yr*Vr
    Scalc = abs(Vr)**2*Yr.conjugate*rede.sb
    Qcalc = Scalc.imag
    
    sn_r = rede.reatores.loc[idx, 'Sn (Mvar)']
    rede.reatores.loc[idx, 'V'] = f'{round(rede.V[barra], 4)} /_ {round(rede.fase[barra]*180/np.pi, 4)}° pu'
    rede.reatores.loc[idx, 'I'] = f'{round(abs(I[barra]*rede.sb/sn_r), 4)} /_ {round(np.angle(I[barra])*180/np.pi, 2)}° pu'
    rede.reatores.loc[idx, 'Q calculado'] = f'{round(Qcalc, 2)} Mvar'

for idx in rede.capacitores.index:
    Yc = rede.capacitores.loc[idx, 'Y']
    barra = rede.capacitores.loc[idx, 'Barra']
    Vc = polar_to_complex(rede.V[barra], rede.fase[barra])
    
    I = Yc*Vc
    Scalc = abs(Vc)**2*Yc.conjugate*rede.sb
    Qcalc = Scalc.imag
    
    sn_c = rede.reatores.loc[idx, 'Sn (Mvar)']
    rede.capacitores.loc[idx, 'V'] = f'{round(rede.V[barra], 4)} /_ {round(rede.fase[barra]*180/np.pi, 4)}° pu'
    rede.capacitores.loc[idx, 'I'] = f'{round(abs(I[barra]*rede.sb/sn_c), 4)} /_ {round(np.angle(I[barra])*180/np.pi, 2)}° pu'
    rede.capacitores.loc[idx, 'Q calculado'] = f'{round(Qcalc, 2)} Mvar'

# Injeção de potência nas barras

Vbarra = pd.Series(polar_to_complex(rede.V, rede.fase), dtype=complex)
I_inj = rede.Ybarra.complex @ Vbarra
S_inj = Vbarra * np.conjugate(I_inj)

# Cálculo da tensão interna nas máquinas

not_starting = rede.motores['Partindo']==False
starting = rede.motores['Partindo']

if all(not_starting):
    for gen in rede.geradores.index:
        barra = rede.geradores.loc[gen, 'Barra']
        i_gen = S_inj[barra]/(Vbarra[barra]*3**0.5)
        vn_kv_gerador = rede.geradores.loc[gen, 'Vn (kV)']
        vn_barra = rede.barras.loc[barra, 'Vn (kV)']
        sn = rede.geradores.loc[gen, 'Sn (MVA)']
        xd = (rede.geradores.loc[gen, 'Xd (%)']/100)*(vn_kv_gerador/vn_barra)**2*rede.sb/sn
        V = Vbarra[barra] + i_gen*xd
        rede.geradores.loc[gen, 'Tensão interna (pu)'] = f'{round(abs(V), 3)} /_ {round(np.angle(V)*180/np.pi, 4)}° pu'
        rede.geradores.loc[gen, 'P (MW)'] = S_inj[barra].real*rede.sb
        rede.geradores.loc[gen, 'Q (Mvar)'] = S_inj[barra].imag*rede.sb
        
# Resultado

if any(starting):
    print('\nFluxo de potência com partida de motor\n')
else:
    print('\nFluxo de potência - regime permanente\n')
    
print('\nBARRAS\n')
print(rede.barras['Tensão calculada'])
if len(rede.linhas)>0:
    print('\nLINHAS\n')
    print(rede.linhas[['De', 'Para', 'V (de)', 'V (para)', 'I (de)', 'I (para)']])
if len(rede.transformadores)>0:
    print('\nTRANSFORMADORES\n')
    print(rede.transformadores[['De', 'Para', 'V (de)', 'V (para)', 'I', 'Sn (MVA)']])
if len(rede.geradores)>0:
    print('\nGERADORES\n')
    print(rede.geradores[['Barra', 'P (MW)', 'Q (Mvar)', 'Tensão interna (pu)']])
if len(rede.reatores)>0:
    print('\nREATORES\n')
    print(rede.reatores[['Barra', 'V', 'Q calculado']])
if len(rede.capacitores)>0:
    print('\nCAPACITORES\n')
    print(rede.capacitores[['Barra', 'V', 'Q calculado']])