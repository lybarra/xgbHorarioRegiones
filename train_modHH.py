# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:30:02 2021

@author: LYbarra
"""
import pyodbc
import pandas as pd
import glob



estaciones =   {10332:'AEROPARQUE AERO',10221:'BAHIA BLANCA AERO',
                10156:'BUENOS AIRES',10270:'COMODORO RIVADAVIA AERO',
                10100:'CORDOBA AERO',10105:'CORDOBA OBSERVATORIO',
                10166:'EZEIZA AERO',10210:'MAR DEL PLATA AERO',
                10131:'MENDOZA AERO',10132:'MENDOZA OBSERVATORIO',
                10227:'NEUQUEN AERO',10111:'PILAR OBS.',
                10489:'RESISTENCIA AERO',10133:'ROSARIO AERO',
                10012:'SALTA AERO',10017:'TUCUMAN AERO'}

regiones={'LIT':10133,
           'NEA':10489,
           'NOA':10017,
           'COM':10227,
           'CUY':10132,
           'GBA':10156,
           'PBS':10221,
           'PBN':10111,
           'PAT':10270,
           'CEN':10105,
           'MDP':10210}

reg_elec=['LIT','NEA','NOA','COM','CUY','GBA','BAS','PAT','CEN']
var_met=['TEMP','HUM','PNM','DD','FF']

path_regelec =r'\\VRANAC\ProSemDi\PRONOSTICO DEMANDA\DB\SMEC_REG_ELEC\*.xls'
path_age=r'\\VRANAC\ProSemDi\PRONOSTICO DEMANDA\DB\SMEC_AGE\*.xls'

#%%

# Importo Demanda por Regiones Eléctricas
appended_data = []

for i in glob.glob(path_regelec, recursive=False):
    df = pd.read_excel(i, skiprows=1)
    df.rename(columns={'Unnamed: 0':'Dia','Unnamed: 1':'Hora'},inplace=True)
    df.drop(df[df['Hora'].isna()].index,inplace=True)
    appended_data.append(df)
    
df = pd.concat(appended_data)


df['Dia']=pd.to_datetime(df.Dia,dayfirst=True)
df['Fecha']=df['Dia'] + pd.to_timedelta(df.Hora, unit='h')
df.drop(['Hora','Dia'],axis=1,inplace=True)
df.set_index('Fecha',inplace=True)
df.sort_index(inplace=True)
df.asfreq(freq='1H')


#%%

# Importo y proceso AGENTES
appended_data = []

for i in glob.glob(path_age, recursive=False):
    df_age = pd.read_excel(i, skiprows=1)
    df_age.rename(columns={'Unnamed: 0':'Dia','Unnamed: 1':'Hora'},inplace=True)
    df_age.drop(df_age[df_age['Hora'].isna()].index,inplace=True)
    appended_data.append(df_age)
    
df_age = pd.concat(appended_data)

df_age['ALUAR']=df_age.ALUAREUA.fillna(0)+df_age.ALUAMAUZ.fillna(0)
df_age.drop(['ALUAREUA','ALUAMAUZ'],axis=1,inplace=True)

df_age['Dia']=pd.to_datetime(df_age.Dia,dayfirst=True)
df_age['Fecha']=df_age['Dia'] + pd.to_timedelta(df_age.Hora, unit='h')
df_age.drop(['Hora','Dia'],axis=1,inplace=True)
df_age.set_index('Fecha',inplace=True)
df_age.sort_index(inplace=True)
df_age.asfreq(freq='1H')
df_age.rename(columns={'ACINVCSZ':'ACINDAR','SDERCA1Z':'SIDERCA'},inplace=True)

# Junto Reg Elec y Agentes
df=df.join(df_age)

#%%
# Importo Feriados
path_fer = r'\\vranac\ProSemDi\PRONOSTICO DEMANDA\INPUT\FERIADOS.xlsx'
df_fer=pd.read_excel(path_fer,index_col=0,parse_dates=True)
df_fer.drop(['Descripcion'],inplace=True,axis=1)
df_fer.set_index(df_fer.index.date,inplace=True)

#%%

## VARIABLES CATEGORICAS CALENDARIO
df['Hora']=df.index.hour
df['Mes']=df.index.month
df['TipoDia']=df.index.dayofweek
df['Dia']=df.index.date

## FERIADOS
df=df.join(df_fer,on='Dia')
df['Feriado'].fillna(0,inplace=True)
df['L.Feriado']=df['Feriado'].shift(24)
df['F.Feriado']=df['Feriado'].shift(-24)
df['F.Feriado'].fillna(0,inplace=True)
df['L.Feriado'].fillna(0,inplace=True)
df['Feriado'].fillna(0,inplace=True)

# DROPEO COLAPSO
# df.drop(df.loc['2019-06-16',].index,axis=0,inplace=True) # #NUEVO

# VALORES REZAGADOS
shifted=[]
for j in reg_elec:
    for i in range(1,25):
        df['L{}.{}'.format(i,j)]=df[j].shift(i)
        shifted.append('L{}.{}'.format(i,j))
    df.drop(df.head(24).index, axis=0,inplace=True)

#%%

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set(rc={'figure.figsize':(17, 8)})
# plt.plot(df.loc['01-01-2021':,'GBA'])

#%%

region='GBA'

estacion=estaciones[regiones[region]].replace(" ","")

appended_data = []


# path =r'\\vranac\ProSemDi\DATOS\SMN\\' + year + '\\**\\rad*.txt'
estacion_path = r'\\vranac\ProSemDi\DATOS\SMN\\**\\ddhh_' + estacion + '*.csv'

shifted=[region]
for i in range(1,25):
        shifted.append('L{}.{}'.format(i,region))

for i in glob.glob(estacion_path, recursive=True):
    df_est = pd.read_csv(i,sep=',',index_col=0)
    appended_data.append(df_est)    
df_est = pd.concat(appended_data)
df_est.set_index(pd.to_datetime(df_est.index),inplace=True)

df_train=df.loc[:,shifted].join(df_est)


#%%

from xgboost.sklearn import XGBRegressor

df_train.dropna(axis=0,inplace=True)

original = ['TEMP','Hora','TipoDia','Mes','Feriado','L.Feriado','F.Feriado']#,'PNM','DD','FF',
feats = original + shifted

since_day = '12-31-2016'
split_day='10-01-2019'
to_day='12-01-2019'

#Separacion TRAIN - TEST
X_train = azul.loc[since_day:split_day,feats]
y_train = azul.loc[since_day:split_day,'Dem']
X_test = azul.loc[split_day:to_day,feats]
y_test = azul.loc[split_day:to_day,'Dem']

print(np.shape(X_train),np.shape(y_train),np.shape(X_test),np.shape(y_test))

from sklearn.model_selection import GridSearchCV

model = XGBRegressor(booster ='gbtree',objective = 'reg:squarederror')

param={
    'learning_rate':[0.1,],
    'max_depth':[5,7],
    'n_estimators':[800],
    'colsample_bytree':[0.9],
    'subsample':[0.9],
    'gamma':[0], #min_split_loss
    'reg_lambda':[30], #reg L2
    'reg_alpha':[0] #reg L1
}

xgb_grid = GridSearchCV(model, param, cv=3 , verbose=False)

#%%
xgb_grid.fit(X_train, y_train, 
        eval_set = [(X_test, y_test)], 
        early_stopping_rounds = 10, 
        verbose=True)

best_param=xgb_grid.best_params_

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

#%%

best_model=XGBRegressor(booster ='gbtree',
                        objective = 'reg:squarederror',
                        **xgb_grid.best_params_)

best_model.fit(X_train, y_train, 
        eval_set = [(X_test, y_test)], 
        early_stopping_rounds = 20, 
        verbose=True)

#%%

fig,ax = plt.subplots()

ax.plot(X_test.index,best_model.predict(X_test), linewidth=1, label='y_pred')
ax.plot(X_test.index,y_test.values,label='y_test')

ax.legend(loc='lower right',frameon=False)
