# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:26:22 2021

@author: LYbarra
"""

import pandas as pd
import numpy as np
from datetime import timedelta,date
from xgboost.sklearn import XGBRegressor
from configparser import ConfigParser

import os
# from sklearn.preprocessing import OneHotEncoder
import joblib

# script_dir = os.path.dirname(__file__)

#leo archivos de configuracion
config = ConfigParser()
privConfig = ConfigParser()
privConfig.read('private.cfg')
config.read('config.cfg')

pathFeriados = privConfig.get('PATHS','pathFeriados')
pathPronosticos = privConfig.get('PATHS', 'pathPronosticos')
pathCovid = privConfig.get('PATHS', 'pathCovid')
pathSalida = privConfig.get('PATHS', 'pathSalida')


featMeteoPron = config.get('FEATURES','featMeteoPron').split(',')
featCat = config.get('FEATURES','featCat').split(',')
featFeriado = config.get('FEATURES','featFeriado').split(',')
regionesElectricas = config.get('REGIONES','regionesElectricas').split(',')


##Datos del dia
hoy=date.today()
dia=hoy.strftime('%d')
mes=hoy.strftime('%m')
anio=hoy.strftime('%y')


################# INPUTS PARA PRONOSTICAR ##################

#importo pronosticos
dfPronosticos = pd.read_excel(pathPronosticos,index_col=0, parse_dates=True)
#variables categoricas calendario
dfPronosticos['TIPODIA'] = dfPronosticos.index.dayofweek
dfPronosticos['MES'] = dfPronosticos.index.month
dfPronosticos.rename(columns={'Hora':'HORA','Dia':'DIA'},inplace=True) #para que quede acorde

# Importo Feriados
dfFer = pd.read_excel(pathFeriados,index_col=0)
dfFer.drop(['Descripcion','Tipo'],inplace=True,axis=1) 
dfFer.set_index(pd.to_datetime(dfFer.index),inplace=True) # sin esto el join arroja error
dfFer.index.names = ['DIA'] #renombro para el join

dfFer = dfFer.asfreq('D',fill_value=0) #completo todo el calendario

dfFer['L.Feriado'] = dfFer['Feriado'].shift(1)
dfFer['F.Feriado'] = dfFer['Feriado'].shift(-1)
dfFer['F.Feriado'].fillna(0,inplace=True) #el día posterior al último dato no se considera feriado
dfFer['L.Feriado'].fillna(0,inplace=True) #el día anteriro al primer dato no se considera feriado

# join con el df de feriados
dfPronosticos = dfPronosticos.join(dfFer,on='DIA')

# modelado del efecto de la cuarentena
dfCovid = pd.read_excel(pathCovid, index_col = 0, parse_dates = True)
dfCovid = dfCovid.asfreq('D')
dfCovid.index.names = ['DIA']

dfPronosticos = dfPronosticos.join(dfCovid, on = 'DIA')
dfPronosticos['COVID'].fillna(0,inplace = True)
#

# defino las features de temperatura al cuadrado y cubo 
tempCols = [col for col in dfPronosticos.columns if 'T2m' in col]

for col in tempCols:
    dfPronosticos[col+'2'] = dfPronosticos[col]**2
    dfPronosticos[col+'3'] = dfPronosticos[col]**3

# uso pronosticos de PBS para BAS
dfPronosticos.columns = dfPronosticos.columns.str.replace('PBS','BAS')



#%%

# MODELOS XGBOOST 
xgbModel=XGBRegressor()
modeloPron = 'GFS'

for region in regionesElectricas: 

    xgbModel.load_model(r'json\{}_xgb.json'.format(region))

    featAux = ['{}_{}_{}'.format(region,modeloPron,i) for i in featMeteoPron]
    #Lista de features, la union de todas
    feats = featAux + featCat + featFeriado
    
    dfPronosticos['pron{}.xgb'.format(region)] = xgbModel.predict(dfPronosticos[feats],validate_features=False).round(0)    



#%%
# archivo de salida

outList = [col for col in dfPronosticos.columns if 'pron' in col]

wr = pd.ExcelWriter(r'{}{}{}{}_reg.xlsx'.format(pathSalida,dia,mes,anio))#,engine='xlsxwriter'

dfPronosticos.loc[:,outList].to_excel(wr,sheet_name='DEMANDA')                  

wr.save()
wr.close()

