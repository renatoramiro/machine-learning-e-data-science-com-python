# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
base = pd.read_csv('credit-data.csv')

base.describe()

# TRATAR DADOS INCONSISTENTES

# Localizar clientes com idades negativas
base.loc[base['age'] < 0]

# Apagar a coluna
base.drop('age', 1, inplace=True)

# Apagar somente os registros com problema
base.drop(base[base.age < 0].index, inplace=True)

# Preencher os valores com a média
base.mean()
base['age'].mean() # valor incorreto porque existem idades negativas
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92

# TRATAR DADOS FALTANTES
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

# Fazendo a divisão entre previsores e classe
# iloc ~> função que fará a divisão
# : ~> pegar todas as linhas
# 1:4 ~> seleção das colunas (income, age, loan), na verdade só vai até o 3
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer # responsável por substituir os valores faltantes
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

# ESCALONAMENTO OU PADRONIZAÇÃO DOS DADOS
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)