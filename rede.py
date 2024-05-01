import pandas as pd
import numpy as np

class Rede:
    def __init__(self, rede: dict, sb: float, f:float = 60) -> None:

        # DataFrames de entrada
        
        self.barras = rede['Barras']
        self.cargas_estaticas = rede['Cargas estáticas']
        self.motores = rede['Motores']
        self.linhas = rede['Linhas']
        self.transformadores = rede['Transformadores']
        self.geradores = rede['Geradores']
        self.reatores = rede['Reatores']
        self.capacitores = rede['Capacitores']
        
        self.sb = sb

        # Aplicação de valores default

        default = {
            'Barras': {'Tipo': 'PQ', 'V (pu)': 1, 'Fase': 0, 'Em serviço':True},
            'Cargas estáticas': {'P (MW)': 0, 'Q (Mvar)': 0, '% P cte': 100, '% I cte': 0, '% Z cte': 0, 'Em serviço':True},
            'Motores': {'P (MW)': 0, 'Q (Mvar)': 0, 'Em serviço':True, 'Partindo':False, 'i_rotor_bloq (pu)':7, 'fp_partida':0.2},
            'Linhas': {'R (ohms/km)': 0, 'C (nF/km)': 0, 'Em serviço':True},
            'Transformadores': {'Tap (De)': 1, 'Fase Tap (De)': 0, 'Tap (Para)': 1, 'Fase Tap (Para)': 0, 'Em serviço':True,},
            'Geradores': {'P (MW)': 0, 'Q (Mvar)': 0, 'Qmin (Mvar)': -np.inf, 'Qmax (Mvar)': np.inf, 'Em serviço':True, 'Ref. p/ transitório':False, 'Tensão interna (pu)':1},
            'Reatores': {'Perdas (MW)': 0, 'Q (Mvar)': 0, 'Em serviço':True},
            'Capacitores': {'Perdas (MW)': 0, 'Q (Mvar)': 0, 'Em serviço':True}
        }
        for df, key in zip([self.barras, self.cargas_estaticas, self.motores, self.linhas, self.transformadores, self.geradores, self.reatores, self.capacitores],
                           ['Barras', 'Cargas estáticas', 'Motores', 'Linhas', 'Transformadores', 'Geradores', 'Reatores', 'Capacitores']):
            df.fillna(default[key], inplace=True)

        # Indexes dos DataFrames que constituem a rede

        self.barras.set_index('ID', inplace=True)
        
        if any(self.geradores['Nome'].duplicated()):
            raise ValueError('Todos os geradores devem ser identificados com um nome único.')
        else:
            self.geradores.set_index('Nome', inplace=True)
            
        for barra in self.barras.index:
            if len(self.geradores[self.geradores['Barra']==barra])>1:
                raise ValueError('É permitido apenas um gerador equivalente por barra.')
        
        # Retirada dos equipamentos que não estão em serviço
        
        for df in (self.barras, self.cargas_estaticas, self.motores, self.linhas, self.transformadores, self.geradores, self.reatores, self.capacitores):
            df.drop(df[df['Em serviço']==False].index, inplace=True)
        
        # Tratamento inicial dos motores

        not_starting = self.motores['Partindo']==False
        starting = self.motores['Partindo']

        self.motores.loc[not_starting, '% P cte'] = 100
        self.motores.loc[not_starting, '% I cte'] = 0
        self.motores.loc[not_starting, '% Z cte'] = 0

        # Configurações para o caso de partida de motores

        if any(starting):
            self.motores.loc[starting, '% P cte'] = 0
            self.motores.loc[starting, '% I cte'] = 0
            self.motores.loc[starting, '% Z cte'] = 100
            fp_partida = self.motores[starting]['fp_partida']
            i_rotor_bloq = self.motores[starting]['i_rotor_bloq (pu)']
            s_partida = self.motores[starting]['Sn (MVA)']*i_rotor_bloq
            self.motores.loc[starting, 'P (MW)'] = s_partida*fp_partida
            self.motores.loc[starting, 'Q (Mvar)'] =  (s_partida**2-self.motores[starting]['P (MW)']**2)**0.5
        
            # Criação das barras de tensão interna dos geradores
            
            for idx in self.geradores.index:
                barra = self.geradores.loc[idx, 'Barra']
                v_interna = self.geradores.loc[idx, 'Tensão interna (pu)']
                vn_kv = self.barras.loc[barra, 'Vn (kV)']
                nome = 'BARRA INTERNA ' + idx
                self.barras = pd.concat(
                    [
                        self.barras,
                        pd.DataFrame(
                            {
                                'Tipo':'PV',
                                'Vn (kV)': vn_kv,
                                'V (pu)': v_interna,
                                'Fase': 0,
                                'Em serviço': True
                            }, index=[nome], columns=self.barras.columns
                        )
                    ], ignore_index=False
                )
                self.geradores.loc[idx, 'Barra'] = nome
        
            # Mudança de barra VT no caso de partida de motores
        
            self.barras.loc[self.barras['Tipo']=='VT', 'Tipo'] = 'PQ'
            gerador_ref = self.geradores[self.geradores['Ref. p/ transitório']==True].index
            if (len(gerador_ref)!=1):
                raise ValueError('Exatamente um gerador deve ser a referência para o cálculo de transitório.')
            v_int_transitorio = self.geradores.loc[gerador_ref[0], 'Tensão interna (pu)']
            barra_ref_transitorio = self.geradores.loc[gerador_ref[0], 'Barra']
            self.barras.loc[barra_ref_transitorio, 'Tipo'] = 'VT'
            self.barras.loc[barra_ref_transitorio, 'V (pu)'] = v_int_transitorio
        
            # Conexão do gerador com a rede via reatância transitória
        
            for idx in self.geradores.index:
                de = 'BARRA INTERNA ' + idx
                para = self.geradores.loc[idx, 'Barra']
                vn_kv_gerador = self.geradores.loc[idx, 'Vn (kV)']
                sn = self.geradores.loc[idx, 'Sn (MVA)']
                xd = self.geradores.loc[idx, 'Xd_t (%)']*vn_kv_gerador**2/(100*sn)
                
                vn_kv_barra = self.barras.loc[barra, 'Vn (kV)']
                self.linhas = pd.concat(
                    [
                        self.linhas,
                        pd.DataFrame(
                            {
                                'De':de, 
                                'Para':barra, 
                                'Nome':f'CONEXÃO {nome}',
                                'R (ohms/km)':0,
                                'X (ohms/km)':xd, 
                                'C (nF/km)':0, 
                                'Vn (kV)':vn_kv_barra, 
                                'Comprimento (km)':1, 
                                'Condutores paralelos':1, 
                                'Em serviço':True
                            }, index=[0]
                        )
                    ], ignore_index=True
                )
        
        # Criação do DataFrame de cargas
        
        self.cargas = pd.concat([self.cargas_estaticas, self.motores], ignore_index=True)
        
        self.cargas['aP'] = self.cargas['% P cte']/100
        self.cargas['bP'] = self.cargas['% I cte']/100
        self.cargas['cP'] = self.cargas['% Z cte']/100
        self.cargas['aQ'] = self.cargas['% P cte']/100
        self.cargas['bQ'] = self.cargas['% I cte']/100
        self.cargas['cQ'] = self.cargas['% Z cte']/100

        # Correções de base
        
        self.cargas['P'] = self.cargas['P (MW)']/sb
        self.cargas['Q'] = self.cargas['Q (Mvar)']/sb
        
        self.linhas['R'] = self.linhas['R (ohms/km)']*self.linhas['Comprimento (km)']*(sb/(self.linhas['Vn (kV)']**2))/self.linhas['Condutores paralelos']
        self.linhas['X'] = self.linhas['X (ohms/km)']*self.linhas['Comprimento (km)']*(sb/(self.linhas['Vn (kV)']**2))/self.linhas['Condutores paralelos']
        self.linhas['Shunt'] = 2*np.pi*f*self.linhas['C (nF/km)']*self.linhas['Condutores paralelos']*1E-9*self.linhas['Vn (kV)']**2/sb
        
        self.geradores['P'] = self.geradores['P (MW)']/sb
        self.geradores['Q'] = self.geradores['Q (Mvar)']/sb
        self.geradores['Qmin'] = self.geradores['Qmin (Mvar)']/sb
        self.geradores['Qmax'] = self.geradores['Qmax (Mvar)']/sb
        
        self.transformadores['Z'] = self.transformadores['Z (%)']*sb/(100*self.transformadores['Sn (MVA)'])
        self.transformadores['R'] = (self.transformadores['Z']**2/(1+(self.transformadores['X/R'])**2))**0.5
        self.transformadores['X'] = (self.transformadores['Z']**2-self.transformadores['R']**2)**0.5
        
        self.reatores['Perdas'] = self.reatores['Perdas (MW)']/sb
        self.reatores['Q'] = self.reatores['Sn (Mvar)']/sb
        
        self.capacitores['Perdas'] = self.capacitores['Perdas (MW)']/sb
        self.capacitores['Q'] = self.capacitores['Sn (Mvar)']/sb
        
        # Cálculo das admitâncias
        
        self.linhas['Y'] = 1/(self.linhas['R']+1j*self.linhas['X'])
        self.linhas['Ysh'] = 1j*self.linhas['Shunt']
        self.transformadores['Y'] = 1 / \
            (self.transformadores['R']+1j*self.transformadores['X'])
        self.reatores['Y'] = -1j*self.reatores['Q']
        self.capacitores['Y'] = 1j*self.capacitores['Q']
        self.transformadores['alfa'] = self.transformadores['Tap (De)']*np.exp(
            1j*self.transformadores['Fase Tap (De)']*np.pi/180)
        self.transformadores['beta'] = self.transformadores['Tap (Para)']*np.exp(
            1j*self.transformadores['Fase Tap (Para)']*np.pi/180)

        # Potências geradas e consumidas nas barras

        self.PL = pd.Series([sum(self.cargas.loc[self.cargas['Barra'] == barra]['P'])
                             for barra in self.barras.index], index=self.barras.index)
        self.QL = pd.Series([sum(self.cargas.loc[self.cargas['Barra'] == barra]['Q'])
                             for barra in self.barras.index], index=self.barras.index)
        self.PG = pd.Series([sum(self.geradores.loc[self.geradores['Barra'] == barra]['P'])
                             for barra in self.barras.index], index=self.barras.index)
        self.QG = pd.Series([sum(self.geradores.loc[self.geradores['Barra'] == barra]['Q'])
                             for barra in self.barras.index], index=self.barras.index)

        # Atributos da rede

        self.barras_PQ = self.barras.loc[self.barras['Tipo'] == 'PQ']
        self.barras_PV = self.barras.loc[self.barras['Tipo'] == 'PV']
        self.barras_VT = self.barras.loc[self.barras['Tipo'] == 'VT']
        self.barras_PQ_ou_PV = pd.concat([self.barras_PQ, self.barras_PV])
        self.barras_VT_ou_PV = pd.concat([self.barras_VT, self.barras_PV])
        self.nPQ = len(self.barras_PQ)
        self.nPV = len(self.barras_PV)
        self.num_equacoes = 2*self.nPQ+self.nPV
        self.num_barras = len(self.barras)
        self.V = self.barras['V (pu)'].copy()
        self.fase = self.barras['Fase']*np.pi/180

        # Matriz admitância calculada na frequência fundamental

        self.Ybarra = Ybarra(self, h=1)


class Ybarra:
    def __init__(self, rede: Rede, h: int):
    
        # Inicialização

        self.complex = pd.DataFrame(
            0, columns=rede.barras.index, index=rede.barras.index, dtype=complex)
        
        for de, para, Ys, Ysh in zip(rede.linhas['De'], rede.linhas['Para'], rede.linhas['Y'], rede.linhas['Ysh']):
            self.complex.at[de, de] += Ys + Ysh/2
            self.complex.at[de, para] += -Ys
            self.complex.at[para, de] += -Ys
            self.complex.at[para, para] += Ys + Ysh/2

        for de, para, Ys, alfa, beta in zip(rede.transformadores['De'], rede.transformadores['Para'],
                                            rede.transformadores['Y'], rede.transformadores['alfa'], rede.transformadores['beta']):
            self.complex.at[de, de] += Ys
            self.complex.at[de, para] += -Ys*alfa/beta
            self.complex.at[para, de] += -Ys*np.conj(alfa/beta)
            self.complex.at[para, para] += Ys*abs(alfa/beta)**2

        for barra, Yr in zip(rede.reatores['Barra'], rede.reatores['Y']):
            self.complex.at[barra, barra] += Yr

        for barra, Yc in zip(rede.capacitores['Barra'], rede.capacitores['Y']):
            self.complex.at[barra, barra] += Yc

        if (h > 1):
            rede.cargas_original['Y'] = rede.cargas_original['P']
            cargas_equivalentes = dict(
                [(barra, sum(rede.cargas_original.loc[rede.cargas_original['Barra'] == barra, 'Y']))
                 for barra in rede.barras.index]
            )
            for barra in cargas_equivalentes.keys():
                self.complex.at[barra, barra] += cargas_equivalentes[barra]

            #self.complex.at[1,1] += 1/(1j*h*0.0001)
            barra_VT = rede.barras_VT.index[0]
            self.complex.drop(barra_VT, inplace=True, axis=0)
            self.complex.drop(barra_VT, inplace=True, axis=1)

        # Módulo e ângulo da matriz admitância

        self.mod = abs(self.complex)
        self.ang = self.complex.applymap(np.angle)