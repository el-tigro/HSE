# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:08:54 2020

@author: El-Tirgo
"""

# загрузка библиотек
import xml.etree.ElementTree as ET
import glob,os
import pandas as pd
from pandas import to_datetime
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import math
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.formula.api import ols
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

# путь к папке
ws = "D:\\Documents\\strahov\\"

########################
# Парсинг xml файлов финансовой отчетности 2012-2016 годов
# (Сбор и конвертация финансовых отчетов из xml в сводную таблицу.)
########################


#название папок с архивами 
folders = ['xml_2012\\', 'xml_2013\\', 'xml_2014\\','xml_2015\\', 'xml_2016\\']
# Формы отчетности
forms = ['1','2']

dfcols = ['id', 'period', 'form','line', 'col', 'value']
df = pd.DataFrame(columns=dfcols)

### Извлечение данных из отчетности за 2011 - 2016 года
for k in folders:
    for i in os.listdir(path=ws+k):
        for j in forms:
            tree = ET.parse(ws+k+i+"\\"+j+".xml")
            root = tree.getroot()
            period=root.attrib.get('period')
            comp_id=root.attrib.get('id')
            for child in root: 
                for child1 in child:
                    for child2 in child1:
                        for child3 in child2:
                            df = df.append(pd.Series([comp_id, period, j, child2.attrib.get('id'), child3.attrib.get('id'), child3.text ], index=dfcols), ignore_index=True)

df.to_csv(ws+'df_all.csv', index = False, sep = ';')
df = pd.read_csv(ws+'df_all.csv', sep = ';')

# извлечение информации по 2011 годe за счет 5-ой колонки в отчетности следующего года
df['col']=pd.to_numeric(df['col'])
df = df[(df['col'] == 4)|((df['col'] == 5)&(df['period'] =='31.12.2012'))]
df['period'] = np.where((df['col'] == 5)&(df['period'] =='31.12.2012'),'31.12.2011',df['period'])
df['id']=pd.to_numeric(df['id'])

df['value'] = df['value'].map(lambda x: ''.join(str(x).split()))
df['value'] = np.where(df['value']!="nan", df['value'], np.nan)
df['value']=pd.to_numeric(df['value'].str.replace(" ",""))

# трансформация таблицы со значениями по всем финансовым факторам в одной колонке в таблицу с множеством колонок различных финансовых переменных
df0=df[['id','period']].drop_duplicates()
for j in df['form'].drop_duplicates():
    line_num=df[df['form']==j]['line'].drop_duplicates()
    for i in line_num:
        df2=df[(df['line']==i)&(df['form']==j)][['id', 'period' ,'value']].drop_duplicates()
        df0 = df0.merge(df2.rename(index=str, columns={'value' : str(j)+"_"+str(i), }),how='left', on = ['id','period'])
del df2

# Извлечение года из даты
df0['year'] = pd.to_numeric(df0['period'].str.replace('31.12.',''))

# сохранение итога
df0.to_csv(ws+'df_all_2.csv', index = False, sep = ';')
del df0

########################
# формирование финансовых относительных факторов 2012-2016 годов
########################

df = pd.read_csv(ws+'df_all_2.csv', sep = ';')

for i in list(df):
    df[i] = df[i].fillna(0)

#   financial data checking
df['1_1300'] = np.where(df['1_1300']<=0,df['1_1000'],df['1_1300'])
df['1_1000']= np.where(df['1_1000']<=0,df['1_1300'],df['1_1000'])
df['1_2200'] = np.where(df['1_2200']<=0, df['1_2000'] - df['1_2100'],df['1_2200'])
df['1_2100'] = np.where(df['1_2100']<=0, df['1_2000'] - df['1_2200'],df['1_2100'])

col = ['2_2120',	'2_2200',	'2_2210',	'2_2220',	'2_2500',	'2_2600',	'2_2610',	'2_2620',	
       '2_2800',	'2_2920',	'2_3100',	'2_3300',	'2_3500',	'2_1400',	'2_1410',	'2_1600',	'2_1800',	'2_1300','2_1120']

for i in col:
    df[i] = -df[i].abs()
    
#### Generate parametres for model
df['fs1']	 = (df['1_2100'] )/(df['1_2200']+df['1_2100'])
df['fs2'] = 	df['1_2100']/(df['1_2210'] + df['1_2220']  - df['1_1230'] - df['1_1240'])
df['fs3']	 = (df['1_2200'] - df['1_2210'] - df['1_2220'] - df['1_2280']) / df['1_2000']
df['fs4']	 = df['1_2200']/df['1_2100']
df['fs5'] = 	df['1_2220']/df['1_1300']
df['fs6']	 = (df['2_2100'] + df['2_1100']) / (df['1_2220'] + df['1_2210'])
#df['fs7']	 = df['1_1300'].map(lambda x: math.log(x))
#df['fs8']	 = np.where((df['2_2100'] + df['2_1100'])>0,(df['2_2100'] + df['2_1100']),np.nan)
#df['fs8']	 = df['fs8'].map(lambda x: math.log(x))


df['pr1']	=	(df['2_3400'] ) / (df['2_1100']  + df['2_1200'] + df['2_1300']  + 
     df['2_1700'] + df['2_1800'] + df['2_2110'] + df['2_2700'] + df['2_2800'] + df['2_2910'] +
     df['2_2920'] + df['2_3200'] + df['2_3300'])
df['pr2']	=	df['2_3400']  / df['1_2100']
df['pr3']	=	df['2_3000'] / df['1_2100']
df['pr4']	=	df['2_3000'] / (df['2_1110'] + df['2_2110'])
df['pr5']	=	df['2_3000'] / df['1_1300']
df['pr6']	=	df['2_3400'] / df['1_1300']

df['unpr1']	=	-(df['2_2200'] + df['2_1400']) /(df['2_2100'] +  df['2_1100'])
df['unpr2']	=	 -(df['2_1600'] + df['2_1800'] + df['2_1700'] + df['2_2220'] + df['2_2600'] +
    df['2_2920'] + df['2_2910'] + df['2_3100'] + df['2_3300'] + df['2_3200'])  / (df['2_1100'] + df['2_2100'] )
df['unpr3']	=	 -(df['2_1400'] + df['2_1500'] + df['2_1600'] + df['2_2200'] + df['2_2600'] + 
    df['2_3100'])  / (df['2_2100'] + df['2_1100'])

df['re1']	=	df['1_1240'] / df['1_2220'] 
#df['re2']	=	(-(df['2_2120'] + df['2_1120']) / (df['2_1100'] + df['2_2100']))
df['re2']	=	(-(df['2_2120'] + df['2_1120']) / (df['2_1110'] + df['2_2110']))
#df['re3']	= - (df['2_1420'] + df['2_2230'])/(df['2_1400'] + df['2_2200'])
df['re3']	= - (df['2_1420'] + df['2_2230'])/(df['2_1410'] + df['2_2210'])


df['inv']	=	(df['1_1140'] + df['1_1270'] ) / ( df['1_2220'] - df['1_1240'])

df['liq1']	=	(df['1_1140'] + df['1_1270']) / (df['1_2200'] - df['1_1230'] - df['1_1240'] - df['1_2280'])
df['liq2']	=	(df['1_1140'] + df['1_1270']) / df['1_1300']
df['solv']	=	-(df['2_1100'] + df['2_2100']) / (df['2_1400'] + df['2_1600'] + df['2_1800'] + df['2_2200'] + df['2_2600'] + df['2_2920'] + df['2_3100'])

param = ['fs1',	'fs2',	'fs3',	'fs4',	'fs5',	'fs6',	#'fs7',	'fs8', 
         'pr1',	'pr2',	'pr3',	'pr4', 'pr5',	'pr6',	'unpr1',	
         'unpr2',	'unpr3',	're1',	're2','re3',	'inv',	'liq1',	'liq2',	'solv']

df['prem'] = df['2_2100'] + df['2_1100']
df['pay'] = df['2_2210'] + df['2_1410']
df['act'] = df['1_1300']

df.to_csv(ws+'df_all_2.csv', index = False, sep = ';')

########################
# парсинг xml-файлов годовой финансовой отчетности 2017 года с новым форматом отчетности
########################

folders = ['xml_2017_msfo\\']
forms = ['39_125','39_126']

dfcols = ['id', 'period', 'form','line', 'col', 'value']
df = pd.DataFrame(columns=dfcols)

for k in folders:
    for i in os.listdir(path=ws+k):
        for j in forms:
            tree = ET.parse(ws+k+i+"\\"+j+".xml")
            root = tree.getroot()
            period=root.attrib.get('period')
            comp_id=root.attrib.get('id')
            for child in root: 
                for child1 in child:
                    if child1.attrib.get('id')!='1c':
                        for child2 in child1:
                            df = df.append(pd.Series([comp_id, period, child.attrib.get('id'), child2.attrib.get('id'), child2[0].attrib.get('id'), child2[0].text ], index=dfcols), ignore_index=True)
                        
                        
df['col']=pd.to_numeric(df['col'])
df = df[df['col']== 4]
df['id']=pd.to_numeric(df['id'])
df['value'] = df['value'].map(lambda x: ''.join(str(x).split()))
df['value'] = np.where(df['value']!="nan", df['value'], np.nan)
df['value']=pd.to_numeric(df['value'].str.replace(" ",""))

df_39 = df[df['form'].isin(['39_125','39_126'])].drop_duplicates()
df_9 = df[df['form'].isin(['9_125','9_126'])].drop_duplicates()


df_39['form'] = np.where(df_39['form'] == '39_125', '1', '2')
forms = ['1','2']


df1 = df_39[['id', 'period', 'form', 'line', 'value']].drop_duplicates()
df0=df1[['id','period']].drop_duplicates()

for j in forms:
    line_num=df1[df1['form']==j]['line'].drop_duplicates()
    for i in line_num:
        df2=df1[(df1['line']==i)&(df1['form']==j)][['id', 'period' ,'value']].drop_duplicates()
        df0 = df0.merge(df2.rename(index=str, columns={'value' : j+"_"+str(i), }),how='left', on = ['id','period'])

df0['year'] = pd.to_numeric(df0['period'].str.replace('31.12.',''))
df0['life'] = np.where(~df0['2_1'].isnull(), 1, 0)

for i in list(df0):
    df0[i] = df0[i].fillna(0)
 
########################
# Формирование относительных финансовых факторов по 2017 году    
########################

df0['fs1']	=	df0['1_51'] / (df0['1_40'] + df0['1_51'])
df0['fs2']	=	df0['1_51'] / (df0['1_30'] + df0['1_33'] - df0['1_9'] - df0['1_11'])
df0['fs3']	=	(df0['1_40'] - df0['1_30'] - df0['1_33'] ) / df0['1_52'] 
df0['fs4']	=	df0['1_40'] / df0['1_51']
df0['fs5']	=	df0['1_33'] / df0['1_23']
df0['fs6']	=	(df0['2_1'] + df0['2_8']) / (df0['1_33'] + df0['1_30'])

df0['prem']	=	df0['2_1'] + df0['2_8']
df0['pay']	=	df0['2_9.1'] + df0['2_2.1']

df0['pr1']	=	(df0['2_30']) / (df0['2_1.1'] + df0['2_5'] + df0['2_6'] +  df0['2_8.1'] + df0['2_22'] 
    + df0['2_12'] + df0['2_13'] + df0['2_27'] + df0['2_28']) 
df0['pr2']	=	df0['2_30'] / df0['1_51']
df0['pr3']	=	df0['2_54'] / df0['1_51']
df0['pr4']	=	df0['2_54'] / (df0['2_1.1'] + df0['2_8.1'])
df0['pr5']	=	df0['2_54'] / df0['1_23']
df0['pr6']	=	df0['2_30'] / df0['1_23']

df0['unpr1']	=	(df0['2_9'] + df0['2_2']) / (df0['2_8'] + df0['2_1'])
df0['unpr2']	=	- (df0['2_4'] + df0['2_5'] + df0['2_6'] + df0['2_9.2'] + df0['2_10'] + df0['2_12']  
                + df0['2_13'] + df0['2_23'] + df0['2_27'] + df0['2_28']) / (df0['2_8'] + df0['2_1'])
df0['unpr3']	=	- (df0['2_2'] + df0['2_3'] + df0['2_4'] + df0['2_9'] + df0['2_10'] +
               df0['2_23']) / (df0['2_8'] + df0['2_1'])
	
df0['re1']	=	df0['1_11'] / df0['1_33']
df0['re2']	=	(df0['2_8.2'] + df0['2_1.2']) / (df0['2_1.1'] + df0['2_8.1'])
df0['re3']	=	(df0['2_2.2'] + df0['2_9.3']) / (df0['2_2'] + df0['2_9.1'])
df0['inv']	=	(df0['1_1'] + df0['1_2'] + df0['1_3'] + df0['1_4'] + df0['1_5']) /  (df0['1_33'] - df0['1_11']) 

df0['liq1']	=	(df0['1_1'] + df0['1_2'] + df0['1_3'] + df0['1_4'] + df0['1_5']) / (df0['1_40'] - df0['1_9'] - df0['1_11'] )
df0['liq2']	=	(df0['1_1'] + df0['1_2'] + df0['1_3'] + df0['1_4'] + df0['1_5']) / df0['1_23']
df0['solv']	=	(df0['2_8'] + df0['2_1']) / (df0['2_2']  + df0['2_4'] + df0['2_6'] + df0['2_9'] + df0['2_10'] + df0['2_13'] + df0['2_23'])

df0['act'] = df0['1_23']

df0.to_csv(ws+'df_2017.csv', sep = ';')

df0 = pd.read_csv(ws+'df_2017.csv', sep = ';')
df17 = df0[['id','year','prem','pay','act'] + param]
df17.to_csv(ws+'df_2017_short.csv', sep = ';')

########################
# Формирование единой базs 2012-2017 года
# и добавление переменных, отвечающих за динамику и дамми-переменных,
# и целевой переменной флага дефолта
########################


# объединение данных за 2011-2016 и 2017 год
df = pd.read_csv(ws+'df_all_2.csv', sep = ';')

df = pd.concat([df[['id','year','prem','pay','act'] + param],df17[['id','year','prem','pay','act'] + param]],ignore_index=True)

# вычисление факторов изменения премий и выплат
df_lag = df[['id','year','prem','pay']].copy()
for i in ['prem','pay']:
    df_lag = df_lag.rename(columns={i:i +'_lag'})   
df_lag['year'] = df['year'] + 1
df = df.merge(df_lag, how = 'left', on = ['id','year'])
for i in ['prem','pay']:
    df['delta_' + i] = (df[i] - df[i + '_lag'])/df[i]
del df_lag

# удаление наблююдений по 2011 и 2010 годам
df1 = df[~df['year'].isin([2010,2011])].drop_duplicates()

# загрузка базы дефолтов
default = pd.read_excel(ws + 'dead_alive.xlsx')
default['def'] = np.where(default['default_date'].isnull(),0,1)
default['year'] = default['default_date'].map(lambda x: x.year) - 1

default = default[['id','year','def']].drop_duplicates()

# проставление маркеров дефолта в нашей выборке
df1 = df1.merge(default[['id','year','def']].rename(columns = {'def':'def_1'}),how='left',on=['id','year'])
df1['def_1'] = np.where(df1['def_1'].isnull(),0,df1['def_1'])

# сохранение полученной выборки
df1.to_csv(ws+'data_all_17_new.csv', index = False)


df1 = pd.read_csv(ws+'data_all_17_new.csv')
df1['pay'] = -df1['pay']

param = ['fs1','fs2','fs3','fs4','fs5','fs6','pr1','pr2', 'pr3', 'pr4', 'pr5','pr6',
 'unpr1', 'unpr2', 'unpr3', 're1', 're2', 're3', 'inv', 'liq1', 'liq2', 'solv', 
 'prem', 'pay', 'delta_prem', 'delta_pay', 'act',
 'insure_sum_new', 'num_zayav', 'num_ureg', 'num_ref', 'num_all_new', 'num_all', 
 'num_ref_div_zayav', 'num_ureg_div_zayav', 'num_ureg_ref_div_zayav', 'num_zayav_div_all',
 'prem_div_insure_sum_new', 'num_all_div_new']

p_add1 = ['delta_prem','div_num_all','div_insure_sum_new','num_ref',
'div_num_zayav','div_act','div_pay','num_ref_div_zayav',
'prem_div_insure_sum_new']

param_new = ['fs1',	'fs2',	'fs4',	'liq1','liq2','pr1','pr2',	'pr4',	'pr6',
             're2',	'solv',	'unpr1',	'unpr2',	'unpr3',	'life']

df1 = df1[['id','year','def_1']+param]

filling_missings = True


# добавление переменной дамми-года
df1 = pd.concat([df1, pd.get_dummies(df1['year'])], axis=1)
param_year = list(pd.get_dummies(df1['year']))

# добавление дамми-переменных видов страховой деятельности
param_kind  = [ 'life', 'medical', 'medical_voluntary', 'avto',
 'exceptlife_other', 'reinsurance_in', 'reinsurance_out',
 'liability', 'mutual'] 
df = pd.read_csv(ws+'kinds_full.csv')

df1 = df1.merge(df[['id','year']+param_kind],how='left',on = ['id','year'])
d = pd.read_csv(ws+'kind.csv')[['id','year','ca']]
df1 = df1.merge(d,how='left',on=['id','year'])
param = param+['ca']
del d,df

# добавление готового региона регистрации из баз спарка
ws1 = ws + 'spark/region/'   
sp = pd.read_excel(ws1 + os.listdir(path=ws1)[0], skiprows=[0,1,2])
sp['year'] = int(os.listdir(path=ws1)[0].replace('.xlsx',''))
    
for i in os.listdir(path=ws1):
    if i != os.listdir(path=ws1)[0]:
        sp_new = pd.read_excel(ws1 + i, skiprows=[0,1,2])
        sp_new['year'] = int(i.replace('.xlsx',''))
        sp = pd.concat([sp,sp_new],ignore_index=True) 
sp = sp.rename(columns={'Год':'year','Регистрационный номер':'id','Регион регистрации':'region'})    
del sp_new

# заполнение оставшихся пропусков по регионам вручную
sp_add = pd.DataFrame({
    'id' : [1,1209,4296,4307,4323,4335,4337,4348,4349,4351],
    'region' : ['Москва', 'Москва', 'Вологодская область', 'Республика Татарстан', 'Москва', 
                'Республика Крым', 'Республика Крым', 'Тюменская область', 'Санкт-Петербург',  'Москва'],
})

# добавление к нашей базе
s = pd.concat([sp[['id','region']],sp_add],ignore_index=True)    
df1 = df1.merge(s[['id','region']].drop_duplicates(),how = 'left',on='id')

df1['reg_mos'] = np.where(df1['region']=='Москва',1,0)

df1.to_csv(ws + 'data_normal2.csv',index=False)



