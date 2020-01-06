# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:25:00 2020

@author: El-Tirgo
"""

df1= pd.read_csv(ws + 'data_normal2.csv')

########################
# Трансформация факторов 
########################

# Заполнение пропусков
if filling_missings:
    for i in list(df1):
        df1[i] =  np.where(df1[i] == float('-inf'), df1[df1[i] != float('-inf')][i].min(),
               np.where(df1[i] == float('inf'), df1[df1[i] != float('inf')][i].max(), 
                        np.where(~df1[i].isnull(), df1[i], 0)))
#        df1[i].fillna(0)#df1[i].median())
else:
    #param = param_new + p_add1
    df1 = df1[['id','year','def_1']+param]
    for i in list(df1):
        df1 = df1[~df1[i].isnull()]
        
# Нормализация финансовых переменных
for i in param:
    if i!='life':
        median = df1[i].median()
        df1[i] = df1[i].fillna(median)
        perc = np.percentile(df1[i], 95)
        slope=-(math.log(1/0.95-1)/(perc-median))
        df1[i] = df1[i].map(lambda x: 1/(1+math.exp(-slope*(x-median))) 
            if -slope*(x-median)<700 else 0 )
    #    mean = df1[i].mean()
    #    std = df1[i].std()
    #    df1[i]=(df1[i]-mean)/std
del median,perc,slope    
   
# Итоговые факторы
param_year = ['2012','2013','2014','2015','2016','2017']
par = param + param_kind + param_year  


############################################
#### WOE - transformation of region name by train data set
############################################


X_train = df1[df1['year']!=2017].drop('def_1', axis = 1).copy()
y_train = df1[df1['year']!=2017][['def_1']].copy()

X_test = df1[df1['year']==2017].drop('def_1', axis = 1).copy()
y_test = df1[df1['year']==2017][['def_1']].copy()

#### WOE - transformation of region name
k = df1[df1['year']!=2017][['region','def_1']]
k['n'] = 1
k = k.rename(columns = {'def_1':'bad'}).groupby('region',as_index=False)['bad','n'].sum()
k['bad'] = np.where(k['bad'] == 0, 0.1, k['bad'])
k['good'] = k['n'] - k['bad']
k['woe_region'] = ((k['good']/k['good'].sum())/(k['bad']/k['bad'].sum())).map(lambda x: math.log(x))

X_train = X_train.merge(k[['region', 'woe_region']], how = 'left', on = 'region')
X_test = X_test.merge(k[['region', 'woe_region']], how = 'left', on = 'region')

X_train = X_train.drop('region', axis = 1)
X_test = X_test.drop('region', axis = 1)


#############
# немонотонно воздействующие переменные будут в дальнейшем трансформированы 
# в переменные вида logit_regression(X_train[i] + X_train[i]^2)
#############
 


