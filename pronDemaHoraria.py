# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:26:22 2021

@author: LYbarra
"""

import pandas as pd
import numpy as np
from datetime import timedelta,date
from xgboost.sklearn import XGBRegressor
import os

from sklearn.preprocessing import OneHotEncoder
import joblib

script_dir = os.path.dirname(__file__)


##Datos del dia
hoy=date.today()
dia=hoy.strftime('%d')
mes=hoy.strftime('%m')
anio=hoy.strftime('%y')

tmin=['RosarioMin','CordobaMin','BahiaBlancaMin',
          'TucumanMin','ResistenciaMin','NeuquenMin','MendozaMin','CapitalMin']
tmax=['RosarioMax','CordobaMax','BahiaBlancaMax',
          'TucumanMax','ResistenciaMax','NeuquenMax','MendozaMax','CapitalMax']


################# INPUTS PARA PRONOSTICAR ##################
in_path=r'\\vranac\ProSemDi\PRONOSTICO DEMANDA\INPUT'
pron2=dia+mes+anio+'.xlsx'
path_pron=os.path.join(in_path,pron2)
data_pron=pd.read_excel(path_pron,sheet_name='TEMP',index_col=0,parse_dates=True)
data_pron=data_pron.asfreq('D')
data_pron=data_pron.astype('float')

data_dem=pd.read_excel(path_pron,sheet_name='DEMA',index_col=0,parse_dates=True)
data_nub=pd.read_excel(path_pron,sheet_name='NUBOSIDAD',index_col=0,parse_dates=True)
data_nub.replace(nub_dic,inplace=True)


fer='FERIADOS.xlsx'
path_fer=os.path.join(in_path,fer)
data_fer=pd.read_excel(path_fer,index_col=0,parse_dates=True)
data_fer.drop(labels=['Descripcion'],axis=1,inplace=True)
data_fer=data_fer.asfreq('D')
data_fer['FeriadoManiana']=data_fer['Feriado'].shift(-1)
data_fer['FeriadoAyer']=data_fer['Feriado'].shift(1)
data_fer.fillna(0,inplace=True)

#Otros Inputs
covid='COVID.xlsx'
path_covid=os.path.join(in_path,covid)
data_covid=pd.read_excel(path_covid,index_col=0,parse_dates=True)
#----------------------------------------------------------

#%%
x_for=pd.merge(data_pron,data_nub,left_index=True,right_index=True,how='left')
x_for=pd.merge(x_for,data_dem,left_index=True,right_index=True,how='left')
x_for=pd.merge(x_for,data_covid,left_index=True,right_index=True,how='left')
x_for=pd.merge(x_for,data_fer,left_index=True,right_index=True,how='left')

x_for['Feriado'].fillna(0,inplace=True)
x_for['FeriadoAyer'].fillna(0,inplace=True)
x_for['FeriadoManiana'].fillna(0,inplace=True)

#%%
#Genero las columnas para rellenar
x_for['MEM']=np.zeros(x_for.shape[0])
x_for['MEM Pico']=np.zeros(x_for.shape[0])
x_for['MEMmin']=np.zeros(x_for.shape[0])

#Genero las variables calendario
x_for['Fecha']=pd.to_datetime(x_for.index)
x_for['Mes'] = x_for['Fecha'].dt.month
x_for['Dia'] = x_for['Fecha'].dt.dayofweek
#Reemplazo los mie,jue,vie (2, 3 y 4) por martes (1)
# x_for['Dia'].replace(to_replace=[2,3,4], value=1,inplace=True)


tmax2, tmin2, tmax3, tmin3 =[],[],[],[]

for i in tmin:
    x_for[i+'2']=x_for[i]**2
    tmin2.append(i+'2')
    x_for[i+'3']=x_for[i]**3
    tmin3.append(i+'3')
    
for i in tmax:
    x_for[i+'2']=x_for[i]**2
    tmax2.append(i+'2')
    x_for[i+'3']=x_for[i]**3
    tmax3.append(i+'3')

#%%
## CARGO EL ONE HOT ENCODER
cat=['Feriado','FeriadoAyer','FeriadoManiana']
cat2=['Dia','NUBOSIDAD','Mes'] 


enc_path = os.path.join(script_dir, 'encoder.joblib')

enc = joblib.load(enc_path)

ohe_cat = enc.get_feature_names(cat2)

enc_df = pd.DataFrame(data = enc.transform(x_for[cat2]).toarray(),
                           columns = ohe_cat,
                           index = x_for.index,
                           dtype = bool)

x_for=x_for.join(enc_df)

#Defino el DataFrame para
x_for_rl=pd.DataFrame(data=x_for.values,index=x_for.index,columns=x_for.columns)
x_for_rl.fillna(0)

#%%

# FEATURES PARA ENTRENAR
misc=['COVID']
feats = tmax + tmin + misc + cat + ohe_cat.tolist() + tmax2 + tmin2 + tmax3 + tmin3
# Genero variables para que sea mas facil aniadirlas a una lista
mem_ene=['L.MEM']
mem_pico=['L.MEMPico']
mem_min=['L.MEMmin']


#%%

# MODELOS XGBOOST 
xgb_mem=XGBRegressor()
xgb_pico=XGBRegressor()
xgb_min=XGBRegressor()


xgb_mem_path = os.path.join(script_dir, 'xgb_mem.json')
xgb_pico_path = os.path.join(script_dir, 'xgb_pico.json')
xgb_min_path = os.path.join(script_dir, 'xgb_min.json')

xgb_mem.load_model(xgb_mem_path)
xgb_pico.load_model(xgb_pico_path)
xgb_min.load_model(xgb_min_path)

# MODELOS REG LINEAL

# rl_pico=LinearRegression()
# rl_mem=LinearRegression()
# rl_min=LinearRegression()

pico_path = os.path.join(script_dir, 'rl_pico.joblib')
rl_pico=joblib.load(pico_path)

mem_path = os.path.join(script_dir, 'rl_mem.joblib')
rl_mem=joblib.load(mem_path)

min_path = os.path.join(script_dir, 'rl_min.joblib')
rl_min=joblib.load(min_path)




#%%


for idx in range(x_for.shape[0]):
    mem_iter=xgb_mem.predict(x_for[feats+mem_pico].iloc[[idx]],validate_features=False)
    x_for['MEM'].iloc[idx]=mem_iter
    
    pico_iter=xgb_pico.predict(x_for[feats+mem_pico].iloc[[idx]],validate_features=False)
    x_for['MEM Pico'].iloc[idx]=pico_iter
    
    min_iter=xgb_min.predict(x_for[feats+mem_pico].iloc[[idx]],validate_features=False)
    x_for['MEMmin'].iloc[idx]=min_iter
    
    if idx+1 in range(x_for.shape[0]):
        x_for['L.MEM'].iloc[idx+1]=mem_iter
        x_for['L.MEMPico'].iloc[idx+1]=pico_iter    
        x_for['L.MEMmin'].iloc[idx+1]=min_iter  
        
        
#%%



for idx in range(x_for_rl.shape[0]):
    mem_iter=rl_mem.predict(x_for_rl[feats+mem_pico].iloc[[idx]])[0]
    x_for_rl['MEM'].iloc[idx]=mem_iter
    
    pico_iter=rl_pico.predict(x_for_rl[feats+mem_pico].iloc[[idx]])[0]
    x_for_rl['MEM Pico'].iloc[idx]=pico_iter
    
    min_iter=rl_min.predict(x_for_rl[feats+mem_pico].iloc[[idx]])[0]
    x_for_rl['MEMmin'].iloc[idx]=min_iter
    
    if idx+1 in range(x_for_rl.shape[0]):
        x_for_rl['L.MEM'].iloc[idx+1]=mem_iter
        x_for_rl['L.MEMPico'].iloc[idx+1]=pico_iter    
        x_for_rl['L.MEMmin'].iloc[idx+1]=min_iter  



#%%
out=pd.DataFrame(x_for.loc[:,['MEM','MEM Pico','MEMmin']],index=x_for.index)

out['MEM_rl']=x_for_rl['MEM']
out['MEM Pico_rl']=x_for_rl['MEM Pico']
out['MEMmin_rl']=x_for_rl['MEMmin']


#Escritura del archivo
path1 = r'\\vranac\ProSemDi\PRONOSTICO DEMANDA\OUTPUT'
path2 = dia +mes + anio + '.xlsx'

path=os.path.join(path1,path2)
wr=pd.ExcelWriter(path)#,engine='xlsxwriter'
out.to_excel(wr,sheet_name='DEMANDA')                  

wr.save()
wr.close()

