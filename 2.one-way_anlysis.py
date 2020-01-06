# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:33:38 2020

@author: El-Tirgo
"""

##############
# Однофакторный анализ
##############

# Gini по отдельным переменным
def Gini_fun(df1,par):
    dfcols_g = ['param','sg', 'gini']
    Gini = pd.DataFrame(columns=dfcols_g)
    for i in par:
        pred = df1[[i]]
        roc_auc = roc_auc_score(df1[['def_1']], pred)
        gini = abs(2*roc_auc - 1)
        sg = np.sign(2*roc_auc - 1)
        Gini = Gini.append(pd.Series([i,sg, gini], index=dfcols_g), ignore_index=True)
    del roc_auc,gini,sg,dfcols_g,pred
    return Gini

Gini = Gini_fun(df1,par)

# T-test по отдельным переменным
dfcols_t = ['param', 'P-value']
Test = pd.DataFrame(columns=dfcols_t)
for k in par:
    X = df1[[k] + ['def_1']].copy()
    st, pval = stats.f_oneway(X[X['def_1']==1][k], X[X['def_1']==0][k])
    Test = Test.append(pd.Series([k, pval], index=dfcols_t), ignore_index=True)

T = Gini.merge(Test,how = 'left',on='param')

writer = pd.ExcelWriter(ws + 'Gini_Test_add1.xlsx')
T.to_excel(writer, index = False)
writer.save()

# переменные, у которых влияние правильное
par_new = ['fs3',	'fs6',	'pr1',	'pr2',	'pr3',	'pr4',	'pr5',	'pr6',	'unpr1',	
           'unpr2',	'unpr3',	're1',	're2',	're3',	'inv',	'liq1',	'liq2',	
           'solv',	'prem',	'delta_prem',	'act',	'num_all_new',	'num_all',	
           'prem_div_insure_sum_new',	'ca']

par_not_u = ['fs3',	'pr2',	'pr3',	'pr5',	'pr6',	'unpr1',	'unpr2',	
             'unpr3',	're3',	'solv',	'prem',	'act',	'num_all_new',	
             'num_all',	'prem_div_insure_sum_new',	'ca']

par_u = ['fs6',	'pr1',	'pr4',	're1',	're2',	'inv',	'liq1',	'liq2',	'delta_prem']

# добавление квадратов переменных
dfcols_g = ['param', 'gini','gini_u']
Gini_u = pd.DataFrame(columns=dfcols_g)
p2 =[]
for i in par_u:
    p2 = p2 + [i +'^1_2']
    X_train[i+'^2'] = X_train[i]**2
    logit_model=sm.Logit(y_train,X_train[[i,i+'^2']])
    result=logit_model.fit()
    #print(result.summary2())
    df1[i+'^2'] = df1[i]*df1[i]
    df1[i +'^1_2'] = result.predict(df1[[i,i+'^2']])
    roc_auc = roc_auc_score(df1[['def_1']], df1[i +'^1_2'])
    gini_u = abs(2*roc_auc - 1)
    Gini_u = Gini_u.append(pd.Series([i,Gini[Gini['param']==i]['gini'].mean(), gini_u], index=dfcols_g), ignore_index=True)
del roc_auc,gini_u,dfcols_g

writer = pd.ExcelWriter(ws + 'Gini_u.xlsx')
Gini_u.to_excel(writer, index = False)
writer.save()

# построение графиков roc-кривых
for j in par_u:
    for i in [j,j +'^1_2']:
        y = df1[['def_1']].copy()
        pred = df1[[i]].copy()
        fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
        roc_auc = auc(fpr, tpr)
        gini = abs(2*roc_auc - 1)
        sg = np.sign(2*roc_auc - 1)
        #Gini = Gini.append(pd.Series([i,sg, gini], index=dfcols_g), ignore_index=True)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic of ' + i)
        plt.legend(loc="lower right")
        plt.savefig(ws + 'pict/' + i + '.png')
        plt.show()
        print(gini)


# Матрица корреляций
M = df1[par_not_u + p2].corr().abs()

writer = pd.ExcelWriter(ws + 'Corr_matrix2.xlsx')
M.to_excel(writer, index = False)
writer.save()

# Удаление сильно коррелируемых переменных
Gini = Gini_fun(df1,par_not_u + p2)
             
dfcols_cor = ['param1','param2','Corr', 'gini1','gini2']
M_cor = pd.DataFrame(columns=dfcols_cor)

p = (par_not_u + p2).copy()
#p = Gini.sort_values(by = 'gini',ascending=False)['param'].copy()

p1 = []
for i in p:
    for j in p:
        if (i!=j) and (M[i][j]>0.75):
            gini1 = Gini[Gini['param']==i]['gini'].max()
            gini2 = Gini[Gini['param']==j]['gini'].max()
            if (gini1>=gini2):
                print(i,j)
                p1 += [j]
                M_cor = M_cor.append(pd.Series([i,j,M[i][j],gini1,gini2], index=dfcols_cor), ignore_index=True)

writer = pd.ExcelWriter(ws + 'Corr_matrix3.xlsx')
M_cor.to_excel(writer, index = False)
writer.save()

p1 = list(set(p1))              
p1.remove('act')

for i in p1:
    p.remove(i)
    
    
df1.to_csv(ws + 'data_normal2.csv')

df1 = pd.read_csv(ws + 'data_normal2.csv')


