# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:30:02 2021

@author: LYbarra
"""
import pyodbc
import pandas as pd
import numpy as np
import glob
from configparser import ConfigParser
from ast import literal_eval #para leer dict de cfg
from datetime import date, timedelta
from scipy.signal import savgol_filter #filtro suavisador media movil

from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split


#Hay dos archivos de cfg, uno para informacion privada y otro para datos generales
#instancio y leo los archivos de conf
config = ConfigParser()
privConfig = ConfigParser()
privConfig.read('private.cfg')
config.read('config.cfg')

#importo de cfg privada
pathFeriados = privConfig.get('PATHS','pathFeriados')
pathEstaciones = privConfig.get('PATHS','pathEstaciones')

#importo de cfg general
estacionesMet = literal_eval(config.get('REGIONES','estacionesMet'))
regionesEstaciones = literal_eval(config.get('REGIONES','regionesEstaciones'))
regionesElectricas = config.get('REGIONES','regionesElectricas').split(',')
acerias = literal_eval(config.get('ACERIAS','acerias'))

featMeteo = config.get('FEATURES','featMeteo').split(',')
featCat = config.get('FEATURES','featCat').split(',')
featFeriado = config.get('FEATURES','featFeriado').split(',')


#Lista de features, la union de todas
feats = featMeteo + featCat + featFeriado 


#defino la conexion a la BD
conx = pyodbc.connect('DSN={};UID={};PWD={}'.format(privConfig['CONN']['DNS'],
                                                  privConfig['CONN']['user'],
                                                  privConfig['CONN']['password']))

#elijo fechas de corte para seleccion de datos
# spDay = date.today()- timedelta(days=90) #si uso tra test split no hace falta
sinceDay = date.today() - timedelta(days=365*3) #entreno con X anios de historia
# ACA PODRIA HACER UN MAX DATE

#%%

#query de demanda sin acerias
sqlDem=r"""
    SELECT TIME.FECHA DIA, TIME.HORA, TOP.RGE_NEMO REGION, SUM(DEM.DEMANDA_REAL_NETA) DEMANDA
    FROM SINGER.DEMANDA_HORARIA DEM
    INNER JOIN SINGER.TIEMPOS TIME
    ON TIME.TIEMPO_ID = DEM.TIEMPO_ID
    INNER JOIN SINGER.TOPOLOGIAS TOP
    ON TOP.TOPOLOGIA_ID = DEM.TOPOLOGIA_ID_AGDEM
    WHERE  TIME.FECHA>{{ts '{} 00:00:00'}} AND 
        DEM.DEMANDA_REAL_NETA>0 AND 
        TOP.AGE_NEMO NOT IN ({})
    GROUP BY TIME.FECHA, TIME.HORA, TOP.RGE_NEMO
    ORDER BY TOP.RGE_NEMO, TIME.FECHA, TIME.HORA
    """.format((sinceDay-timedelta(days=1)).strftime("%Y-%m-%d"),
                str(list(acerias.keys()))[1:-1])

qryDem = pd.read_sql(sqlDem,conx)

# armo el index como timestamp, luego reemplazo dia y hora para reemplazar la H24 por 0
qryDem.HORA = qryDem.HORA.astype('int')
qryDem['Fecha'] = pd.to_datetime(qryDem.DIA) + pd.to_timedelta(qryDem.HORA, unit='h')
qryDem.set_index('Fecha',inplace=True)
qryDem.HORA = qryDem.index.hour
qryDem.DIA = qryDem.index.date.astype('datetime64[ns]')


#%%

# Importo Feriados
dfFer = pd.read_excel(pathFeriados,index_col=0)
dfFer.drop(['Descripcion','Tipo'],inplace=True,axis=1) #el tipo de feriado se podría inplementar a futuro
dfFer.set_index(pd.to_datetime(dfFer.index).date,inplace=True)
dfFer = dfFer.asfreq('D') 
dfFer.index.names = ['DIA'] #renombro para join

## rezagos de feriados
dfFer['Feriado'].fillna(0,inplace=True)
dfFer['L.Feriado'] = dfFer['Feriado'].shift(1)
dfFer['F.Feriado'] = dfFer['Feriado'].shift(-1)
dfFer['F.Feriado'].fillna(0,inplace=True) #el día posterior al último dato no se considera feriado
dfFer['L.Feriado'].fillna(0,inplace=True) #el día anteriro al primer dato no se considera feriado

# join con el df de feriados
qryDem = qryDem.join(dfFer,on='DIA')


#tratamiento del efecto de la cuarentena
qryDem['COVID'] = np.zeros(len(qryDem.index))
qryDem.loc['2020-03-20':'2020-05-05','COVID'] += 1 #Dias fuerte de cuarentena
qryDem.loc['2020-05-06':'2020-05-25','COVID'] += 0.8 #despues se fue suavizando
qryDem.loc['2020-05-26':'2020-11-01','COVID'] += 0.5 #se flexibiliza mas hasta normalidad


#%%
# ploteo para chequear
# plt.plot(qryDem[(qryDem['REGION']=='GBA')].loc['01-01-2020':,'DEMANDA'])



#%%


for region in regionesElectricas:

    #el nombre de la estacion para la region elegida:
    estacion = estacionesMet[regionesEstaciones[region]].replace(" ","")
    
    #recursivamente abro los csv con los datos meteorológicos de esa estacion
    appended_data = []
    pathEstacion = pathEstaciones + estacion + '*.csv'
    
    for i in glob.glob(pathEstacion, recursive=True):
        dfEst = pd.read_csv(i,sep=',',index_col=0)
        appended_data.append(dfEst)
    dfEst = pd.concat(appended_data)
    dfEst.set_index(pd.to_datetime(dfEst.index),inplace=True)
    
    
    dfDemanda = qryDem[(qryDem['REGION']==region)] #extraigo demanda para la region actual

    dfDemanda.dropna(axis=0,inplace=True)
    
    dfDemanda = dfDemanda.join(dfEst,how='inner') #join con datos meteorologicos
    
    #aplico un filtro de suavizado de las var meteorologicas
    smVar = ['TEMP','PNM','DD','FF','HUM']
    for i in smVar:
        dfDemanda[i] = savgol_filter(dfDemanda[i].values,11,2)
    
    #potencia de las temperaturas
    dfDemanda['TEMP2'] = dfDemanda['TEMP']**2
    dfDemanda['TEMP3'] = dfDemanda['TEMP']**3
    
    dfDemanda['MES'] = dfDemanda.index.month
    dfDemanda['TIPODIA'] = dfDemanda.index.dayofweek 
    

    # DROPEO COLAPSO SI ESTA EN EL DATASET
    try:
        dfDemanda.drop(dfDemanda.loc['2019-06-16',].index,axis=0,inplace=True) 
    except: 
        pass
    
    # Train Test split
    X_train, X_test, y_train, y_test = train_test_split(dfDemanda.loc[sinceDay:,feats],
                                                        dfDemanda.loc[sinceDay:,'DEMANDA'],
                                                        test_size=0.15,random_state=42)
    
    #ajusto el GridSearch SOLO para la primer región y luego uso los mismos hyperp
    if region == regionesElectricas[0]:
        #Instancio el modelo para el GridSearch
        model = XGBRegressor(booster = 'gbtree',objective = 'reg:squarederror', n_jobs = 8)
        
        param={
            'learning_rate':[0.1],
            'max_depth':[5,7],
            'n_estimators':[300],
            'colsample_bytree':[0.7,0.9],
            'subsample':[0.7,0.9],
            'gamma':[0], #min_split_loss
            'reg_lambda':[30,50], #reg L2
            'reg_alpha':[0,10] #reg L1
        }
        
        xgb_grid = GridSearchCV(model, param, cv=2)
        
        #AJUSTO LOS MEJORES PARAMETROS CON GRID SEARCH
        xgb_grid.fit(X_train, y_train, 
                eval_set = [(X_test, y_test)], 
                early_stopping_rounds = 10, 
                verbose=False)
        
        bestParams = xgb_grid.best_params_ #me guardo el dict
        
        print(xgb_grid.best_score_)
        print(xgb_grid.best_params_)
        #una vez que elegi los hyper, subo el n_estimator, y podría bajar el lr
        bestParams['n_estimators'] = 1000
        #bestParams['learning_rate'] = 0.01
    
    #fiteo el modelo definitivo
    bestModel = XGBRegressor(booster ='gbtree',
                            objective = 'reg:squarederror',
                            n_jobs = -1,
                            **bestParams)
    
    bestModel.fit(X_train, y_train, 
                    eval_set = [(X_test, y_test)], 
                    early_stopping_rounds = 20, 
                    verbose=True)
    
    bestModel.save_model(r'json\{}_xgb.json'.format(region))

#%%

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(rc={'figure.figsize':(17, 8)})

# model = XGBRegressor() 
# model.load_model('PAT_xgb.json')

# fig,ax = plt.subplots(6,1,sharex=True)

# start = '2021-08-10'
# end = '2021-08-17'

# idx = dfDemanda.loc[start:end,].index
# yPred = model.predict(dfDemanda.loc[start:end,feats])
# yTrue = dfDemanda.loc[start:end,'DEMANDA']

# ax[0].plot(idx, yPred, linewidth=1, label='y_pred')
# ax[0].plot(idx, yTrue,label='y_true')
# ax[0].legend(loc='lower right',frameon=False)

# ax[1].plot(dfDemanda.loc[start:end,'TEMP'])
# ax[2].plot(dfDemanda.loc[start:end,'FF'])
# ax[3].plot(dfDemanda.loc[start:end,'DD'])
# ax[4].plot(dfDemanda.loc[start:end,'HUM'])
# ax[5].plot(dfDemanda.loc[start:end,'PNM'])
#%%




#%%




