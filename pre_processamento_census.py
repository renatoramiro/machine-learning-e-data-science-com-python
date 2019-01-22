#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:19:04 2019

@author: renato
"""

import pandas as pd

base = pd.read_csv('census.csv')

# age - numérica discreta
# workclass - categórica nominal
# final weight - numerica contínua
# education - categórica ordinal
# education - numérica discreta
# marital status - categórica nominal
# occupation - categórica nominal
# relationship - categórica nominal
# race - categórica nominal
# sex - categórica nominal

# TRANSFORMAÇÃO DE VARIÁVEIS CATEGÓRICAS
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# LabelEncoder - classe responsável pela transformação de string em números
# OneHotEncoder - 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_previsores = LabelEncoder()
#labels = labelencoder_previsores.fit_transform(previsores[:, 1])

previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1]) #workclass
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3]) #education
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5]) #marital-status
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6]) #occupation
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7]) #relationship
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8]) #race
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9]) #sex
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13]) #native-country

onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# ESCALONAMENTO DE ATRIBUTOS
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)