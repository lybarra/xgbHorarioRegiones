# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:30:02 2021

@author: LYbarra
"""
import pyodbc
import pandas as pd
import glob
from configparser import ConfigParser
from ast import literal_eval #para leer dict de cfg
from datetime import date, timedelta
from scipy.signal import savgol_filter #filtro suavisador

from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Si quisiera plotear
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(rc={'figure.figsize':(17, 8)})

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


varMeteo=['TEMP','HUM','PNM','DD','FF']


#instancio la conexion a la BD
conx=pyodbc.connect('DSN={};UID={};PWD={}'.format(privConfig['CONN']['DNS'],
                                                  privConfig['CONN']['user'],
                                                  privConfig['CONN']['password']))

#elijo fechas de corte para seleccion de datos y para train-test split
spDay = date.today()- timedelta(days=90) #si uso tra test split no hace falta
sinceDay = date.today() - timedelta(days=365*3) #entreno con X anios de historia
# ACA DEBERIA HACER UN MAX DATE

#%%

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


qryDem.HORA = qryDem.HORA.astype('int')
qryDem['Fecha'] = pd.to_datetime(qryDem.DIA) + pd.to_timedelta(qryDem.HORA, unit='h')
qryDem.set_index('Fecha',inplace=True)


#%%
# Importo Feriados
dfFer = pd.read_excel(pathFeriados,index_col=0)
dfFer.drop(['Descripcion','Tipo'],inplace=True,axis=1) #el tipo de feriado se podría inplementar a futuro
dfFer.set_index(pd.to_datetime(dfFer.index),inplace=True)


#%%
# ploteo para chequear
# plt.plot(qryDem[(qryDem['REGION']=='GBA')].loc['01-01-2020':,'DEMANDA'])

#%%

region='GBA'

estacion=estacionesMet[regionesEstaciones[region]].replace(" ","")

appended_data = []

pathEstacion = pathEstaciones + estacion + '*.csv'


for i in glob.glob(pathEstacion, recursive=True):
    dfEst = pd.read_csv(i,sep=',',index_col=0)
    appended_data.append(dfEst)
dfEst = pd.concat(appended_data)
dfEst.set_index(pd.to_datetime(dfEst.index),inplace=True)


dfDemanda = qryDem[(qryDem['REGION']==region)] #extraigo demanda para la region actual
dfDemanda['dDEMANDA'] = dfDemanda['DEMANDA'].diff(1)
dfDemanda.dropna(axis=0,inplace=True)

dfDemanda = dfDemanda.join(dfEst,how='inner') #join con datos meteorologicos

#aplico un filtro de suavizado de las var meteorologicas
smVar=['TEMP','PNM','DD','FF','HUM']
for i in smVar:
    dfDemanda[i]=savgol_filter(dfDemanda[i].values,11,2)



dfDemanda['TEMP2'] = dfDemanda['TEMP']**2
dfDemanda['TEMP3'] = dfDemanda['TEMP']**3



dfDemanda['MES'] = dfDemanda.index.month
dfDemanda['TIPODIA'] = dfDemanda.index.dayofweek 

dfDemanda = dfDemanda.join(dfFer)


## FERIADOS
dfDemanda['Feriado'].fillna(0,inplace=True)
dfDemanda['L.Feriado']=dfDemanda['Feriado'].shift(24)
dfDemanda['F.Feriado']=dfDemanda['Feriado'].shift(-24)
dfDemanda['F.Feriado'].fillna(0,inplace=True) #el día posterior al último dato no se considera feriado
dfDemanda['L.Feriado'].fillna(0,inplace=True) #el día anteriro al primer dato no se considera feriado

# # VALORES REZAGADOS
# shifted=[]

# for i in range(1,4):
#     dfDemanda['L{}.Dem'.format(i)]=dfDemanda['DEMANDA'].shift(i)
#     shifted.append('L{}.Dem'.format(i))
# dfDemanda.drop(dfDemanda.head(4).index, axis=0,inplace=True)

#%%

#Lista de features 
featMeteo = ['TEMP','TEMP2','TEMP3','PNM','DD','FF','HUM']
featCat = ['HORA', 'TIPODIA','MES']
featFeriado = ['Feriado', 'L.Feriado','F.Feriado']

feats = featMeteo + featCat + featFeriado #+ shifted


# DROPEO COLAPSO SI ESTA EN EL DATASET
try:
    dfDemanda.drop(dfDemanda.loc['2019-06-16',].index,axis=0,inplace=True) 
except: 
    pass

#Separacion TRAIN - TEST
#Parte vieja
# X_train = dfDemanda.loc[sinceDay:spDay,feats]
# y_train = dfDemanda.loc[sinceDay:spDay,'DEMANDA']
# X_test = dfDemanda.loc[spDay:,feats]
# y_test = dfDemanda.loc[spDay:,'DEMANDA']

X_train, X_test, y_train, y_test = train_test_split(dfDemanda.loc[since_day:,feats],
                                                    dfDemanda.loc[since_day:,'DEMANDA'],
                                                    test_size=0.15,random_state=42)


#Instancio el modelo para el GridSearch
model = XGBRegressor(booster ='gbtree',objective = 'reg:squarederror')

param={
    'learning_rate':[0.1],
    'max_depth':[5,7],
    'n_estimators':[400],
    'colsample_bytree':[0.7,0.9],
    'subsample':[0.7,0.9],
    'gamma':[0], #min_split_loss
    'reg_lambda':[30,50,100], #reg L2
    'reg_alpha':[0,10] #reg L1
}

xgb_grid = GridSearchCV(model, param, cv=3 , verbose=False)

#%%

#AJUSTO LOS MEJORES PARAMETROS CON GRID SEARCH
xgb_grid.fit(X_train, y_train, 
        eval_set = [(X_test, y_test)], 
        early_stopping_rounds = 10, 
        verbose=True)

bestParams = xgb_grid.best_params_ #me guardo el dict

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

#%%

bestParams['n_estimators']=1000
bestParams['learning_rate']=0.01

bestModel=XGBRegressor(booster ='gbtree',
                        objective = 'reg:squarederror',
                        **bestParams)

bestModel.fit(X_train, y_train, 
        eval_set = [(X_test, y_test)], 
        early_stopping_rounds = 20, 
        verbose=True)

bestModel.save_model(region,'_xgb.json')

#%%
# 
# fig,ax = plt.subplots()

# start='2021-08-01 00'
# end='2021-08-08'

# idx = X_test.loc[start:end,].index
# yPred = best_model.predict(X_test.loc[start:end,])
# yTrue = y_test.loc[start:end,].values

# ax.plot(idx, yPred, linewidth=1, label='y_pred')
# ax.plot(idx,yTrue,label='y_test')

# ax.legend(loc='lower right',frameon=False)

#%%

plt.scatter(X_test.loc[start:end,'MES'],(yTrue-yPred))

#%%

# dfDemanda.loc[start,'DEMANDA']

# yPred2 = yPred

# yPred2[0] = yPred[0] + dfDemanda.loc[y_train.index[-2],'DEMANDA']

# for i in range(1,len(yPred)):
#     yPred2[i]=yPred2[i]+yPred2[i-1]


# plt.plot(idx, yPred2, linewidth=1, label='y_pred')
# plt.plot(idx, dfDemanda.loc[start:end,'DEMANDA'],label='y_test')
# plt.legend(loc='lower right',frameon=False)





