# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:34:34 2020

@author: El-Tirgo
"""



########################
# Создание моделей
########################

p = ['fs3','pr2','unpr1','unpr2','re3','solv','act','num_all','prem_div_insure_sum_new',
 'ca','fs6^1_2','pr1^1_2','pr4^1_2','re1^1_2','re2^1_2','liq1^1_2','liq2^1_2','delta_prem^1_2']

param_year = ['2012','2013','2014','2015','2016']

param_kind = [# 'address',
 'life', 'medical', 'medical_voluntary', 'avto',
 'exceptlife_other', 'reinsurance_in', 'reinsurance_out',
 'liability', 'mutual'] 

region = ['reg_mos']

par = p + param_kind + region + param_year + ['const'] #+ ['2012-2013','2014-2015']# 

df1 = pd.read_csv(ws + 'data_normal2.csv').drop('Unnamed: 0',axis = 1)

#разделение выборки на train и test
df1 = add_constant(df1)

X_train = df1[df1['year']!=2017][par].copy()
y_train = df1[df1['year']!=2017][['def_1']].copy()

X_test = df1[df1['year']==2017][par].copy()
y_test = df1[df1['year']==2017][['def_1']].copy()

def logit_fun(p):
    logit_model=sm.Logit(y_train,X_train[p])
    result=logit_model.fit()
    res = result.summary2()
    #print(res)
    predicted = result.predict(X_train[p])
    roc_auc = roc_auc_score(y_true=y_train, y_score=predicted)
    gini_tr = abs(2*roc_auc - 1)
    #print('gini_train = ', gini_tr)
    predicted = result.predict(X_test[p])
    roc_auc = roc_auc_score(y_true=y_test, y_score=predicted)
    gini_ts = abs(2*roc_auc - 1)
    #print('gini_test = ', gini_ts)
    
    d = pd.DataFrame({'pred':predicted,'fact':y_test['def_1']})
    d['pred1']=np.where(d['pred']>0.2,1,0)
    d['n']=1
    d1 = pd.DataFrame({'0_fact':[d[(d['pred1']==0)&(d['fact']==0)]['n'].sum(),d[(d['pred1']==1)&(d['fact']==0)]['n'].sum()],
                              '1_fact':[d[(d['pred1']==0)&(d['fact']==1)]['n'].sum(),d[(d['pred1']==1)&(d['fact']==1)]['n'].sum()]})
    print(d1)
    
    return res, gini_tr, gini_ts


def logit_fun_1(p):
    clf = LogisticRegression()#LogisticRegressionCV(random_state=1)
    clf.fit(X_train[p], y_train)
    predicted = clf.predict_proba(X_train[p])
    roc_auc = roc_auc_score(y_true=y_train, y_score=predicted)
    gini_tr = abs(2*roc_auc - 1)
    
    return gini_tr, clf.densify(), clf.get_params()

gini_tr, densify, params = logit_fun_1(par)


def roc_pic(p):
    logit_model=sm.Logit(y_train,X_train[p])
    result=logit_model.fit()
    predicted = result.predict(X_train[p])
    fpr_tr, tpr_tr, _ = metrics.roc_curve(y_train, predicted, pos_label=1)
    roc_auc_tr = auc(fpr_tr, tpr_tr)
    gini_tr = abs(2*roc_auc_tr - 1)
    predicted = result.predict(X_test[p])
    fpr_ts, tpr_ts, _ = metrics.roc_curve(y_test, predicted, pos_label=1)
    roc_auc_ts = auc(fpr_ts, tpr_ts)
    gini_ts = abs(2*roc_auc_ts - 1)
    #Gini = Gini.append(pd.Series([i,sg, gini], index=dfcols_g), ignore_index=True)
    plt.figure()
    lw = 2
    plt.plot(fpr_tr, tpr_tr, color='darkorange',
             lw=lw, label='Gini train = %0.2f' % gini_tr)
    plt.plot(fpr_ts, tpr_ts, color='orange',
             lw=lw, label='Gini test = %0.2f' % gini_ts)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic of ' + i)
    plt.legend(loc="lower right")
    plt.savefig(ws+ "ROC.png")
    plt.show()
    
    
    
    
    #print(gini)

# Логит-модель от всех переменных
res, gini_tr, gini_ts = logit_fun(par)
roc_pic(par)



#clf = LogisticRegression().fit(X_train, y_train)
#clf.predict(X_test)
# Логит-модель от значимых переменных
par_new = ['pr2','unpr1','act','fs6^1_2','pr4^1_2','re1^1_2','re2^1_2','liq2^1_2','delta_prem^1_2',
           'life','medical','2012','2013'] +['const']

res, gini_tr, gini_ts = logit_fun(par_new)
print(res)
print('gini_train = ', gini_tr)
print('gini_test = ', gini_ts)
roc_pic(par_new)
# vif
def vif(p):
    X = add_constant(X_train[p])
    return pd.Series([variance_inflation_factor(X.values, i) 
                   for i in range(X.shape[1])], 
                  index=X.columns)

v = vif(par_new)

writer = pd.ExcelWriter(ws + 'vif.xlsx')
v.to_excel(writer)
writer.save()

# RFECV
def RFE_feature_selection(clf_lr,n):
    rfecv = RFECV(estimator=clf_lr, step=1,verbose=0, cv=StratifiedKFold(n), scoring='roc_auc')
    rfecv.fit(X_train, y_train)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    f, ax = plt.subplots(figsize=(14, 9))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (roc_auc)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.savefig(ws+ "fselect.png")
    plt.show()
    mask = rfecv.get_support()
    X = X_train.ix[:, mask]
    par_new = list(X)
    print(rfecv.grid_scores_)
    res, gini_tr, gini_ts = logit_fun(par_new)
    print(res)
    print('gini_train = ', gini_tr)
    print('gini_test = ', gini_ts)
    return list(X)

clf_lr = LogisticRegression()#LogisticRegressionCV(random_state=1)
clf_lr.fit(X_train, y_train)

RFE_feature_selection(clf_lr,5)
RFE_feature_selection(clf_lr,2)

# Перебор различных подмножеств факторов
# 8 и 9
fl = list(itertools.combinations(p,8))+list(itertools.combinations(p,9))
flk = list(itertools.combinations(param_kind,1))+list(itertools.combinations(param_kind,2))
fly = [['2012','2013', 'const']]#fly1 + fly2

dfcols_g = ['param','num', 'gini']
M = pd.DataFrame(columns=dfcols_g)

n=0
for i1 in fl:
#    for i2 in flk:
#        for i3 in fly:
    i = list(i1)#+i2+i3
    n=n+1
    X = X_train[i].copy()
    y = y_train[['def_1']].copy()
    logit_model=sm.Logit(y,X[i])
    result=logit_model.fit()
    #print(result.summary2())
    predicted = result.predict(X[i])
    roc_auc = roc_auc_score(y_true=y_train, y_score=predicted)
    gini = abs(2*roc_auc - 1)
    print('gini = ', gini)
    print('num = ', n)
    p1 = ';'
    for j in list(np.arange(len(i))):
        p1 = p1 + i[j] + ";"
    M = M.append(pd.Series([p1, len(i),gini], index=dfcols_g), ignore_index=True)
    
M.to_csv(ws + 'M_8_or_9_param.csv',index=False)

(M['gini']>0.425).sum()
M[M['gini']>0.35]['param']


dfcols_g1 = ['param','num', 'gini1','gini2']
M1 = pd.DataFrame(columns=dfcols_g1)

n=0
for l in M[M['gini']>0.425]['param']:
    i1 = l.split(';')
    i1.remove('')
    i1.remove('')
    gini1 = M[(M['param']==l)]['gini'].mean()
    for i2 in flk:
        for i3 in fly:
            i = i1+i2+i3
            n=n+1
            X = X_train[i].copy()
            y = y_train[['def_1']].copy()
            logit_model=sm.Logit(y,X[i])
            result=logit_model.fit()
            #print(result.summary2())
            predicted = result.predict(X[i])
            roc_auc = roc_auc_score(y_true=y_train, y_score=predicted)
            gini = abs(2*roc_auc - 1)
            print('gini = ', gini)
            print('num = ', n)
            p = ';'
            for j in list(np.arange(len(i))):
                p1 = p1 + str(i[j]) + ";"
            M1 = M1.append(pd.Series([p1, len(i), gini1, gini], index=dfcols_g1), ignore_index=True)

M1.to_csv(ws + 'M_8_or_9_param_with_kind_year.csv',index=False)


dfcols_g2 = ['param','num', 'gini_tr','gini_ts']
M2 = pd.DataFrame(columns=dfcols_g2)


(M1['gini2']>0.55).sum()

for l in M1[M1['gini2']>0.55]['param']:
    l1 = l.split(';')
    l1.remove('')
    l1.remove('')
    X = X_train[l1].copy()
    y = y_train[['def_1']].copy()
    logit_model=sm.Logit(y,X[l1])
    result=logit_model.fit()
    predicted = result.predict(X_test[l1])
    roc_auc = roc_auc_score(y_true=y_test, y_score=predicted)
    gini = abs(2*roc_auc - 1)
    print('gini = ', gini)
    M2 = M2.append(pd.Series([l, len(l1), M1[M1['param']==l]['gini2'].mean(), gini], index=dfcols_g2), ignore_index=True)


l = ';pr2;unpr1;num_all;prem_div_insure_sum_new;fs6^1_2;pr1^1_2;pr4^1_2;liq2^1_2;delta_prem^1_2;life;medical;2012;2013;const;'
l1 = l.split(';')
l1.remove('')
l1.remove('')
l1.remove('prem_div_insure_sum_new')
l1.remove('pr1^1_2')
res, gini_tr, gini_ts = logit_fun(l1)
print(res)
print('gini_train = ', gini_tr)
print('gini_test = ', gini_ts)


#### 5,6,7
fl = list(itertools.combinations(p,5))+list(itertools.combinations(p,6))+list(itertools.combinations(p,7))
flk = ['life','medical']
fly = ['2012','2013', 'const']#fly1 + fly2

def pereb(fl,flk,fly):
    dfcols_g = ['param','num', 'gini_tr','gini_ts']
    M3 = pd.DataFrame(columns=dfcols_g)
    
    n=0
    for i1 in fl:
    #    for i2 in flk:
    #        for i3 in fly:
        i = list(i1) + flk + fly#+i2+i3
        n=n+1
        res, gini_tr, gini_ts = logit_fun(i)
        print('gini_ts = ', gini_ts)
        print('num = ', n)
        p = ';'
        for j in list(np.arange(len(i))):
            p = p + i[j] + ";"
        M3 = M3.append(pd.Series([p, len(i),gini_tr, gini_ts], index=dfcols_g), ignore_index=True)
    return M3

M3.to_csv(ws + 'M_5_or_6_or_7_param_with_kind_year.csv',index=False)

writer = pd.ExcelWriter(ws + 'M_5_or_6_or_7_param_with_kind_year.xlsx')
M3.to_excel(writer)
writer.save()

for l in M4[(M4['gini_tr']>=0.555)&(M4['gini_ts']>0.60)&(M4['num']==13)]['param']:
    #l = ';pr2;unpr1;num_all;fs6^1_2;pr4^1_2;re1^1_2;delta_prem^1_2;life;medical;2012;2013;const;'
    l1 = l.split(';')
    l1.remove('')
    l1.remove('')
    
    res, gini_tr, gini_ts = logit_fun(l1)
    print(res)
    print('gini_train = ', gini_tr)
    print('gini_test = ', gini_ts)
roc_pic(l1)

fl = list(itertools.combinations(p,8))+list(itertools.combinations(p,9))
M4 = pereb(fl,flk,fly)

writer = pd.ExcelWriter(ws + 'M_8_or_9_param_with_kind_year.xlsx')
M3.to_excel(writer)
writer.save()

M4.to_csv(ws + 'M_8_or_9_param_with_kind_year.csv',index=False)

M3[(M3['gini_tr']>0.55)].max().shape

M31 = M3[(M3['gini_tr']>0.55)&(M3['gini_ts']>0.56)].shape

writer = pd.ExcelWriter(ws + 'M_8_or_9_param_with_kind_year_short.xlsx')
M41.to_excel(writer)
writer.save()

M41 = M4[(M4['gini_tr']>=0.555)&(M4['gini_ts']>0.61)&(M4['num']==13)].shape
##########################################
# Машинное обучение
##########################################

# DecisionTree
DecisionTree = DecisionTreeClassifier(criterion='gini', max_depth = 20, min_samples_leaf = 1,min_samples_split=4 , random_state=0)
result=DecisionTree.fit(X_train[par], y_train)
print(DecisionTree.feature_importances_)
predicted = result.predict(X_train[par])
roc_auc = roc_auc_score(y_true=y_train, y_score=predicted)
print('gini_tr = ', abs(2*roc_auc - 1))
predicted = result.predict(X_test[par])
roc_auc = roc_auc_score(y_true=y_test, y_score=predicted)
print('gini_test = ', abs(2*roc_auc - 1))
## 91.2,  22.5



# svm 
clf = svm.SVC(C=2, kernel = 'linear', probability=True, class_weight = 'balanced')
result=clf.fit(X_train[par], y_train['def_1'])
predicted = pd.DataFrame(result.predict_proba(X_train[par]))[1]
roc_auc = roc_auc_score(y_true=y_train, y_score=predicted)
print('gini_tr = ', abs(2*roc_auc - 1))
predicted = pd.DataFrame(result.predict_proba(X_test[par]))[1]
roc_auc = roc_auc_score(y_true=y_test, y_score=predicted)
print('gini_test = ', abs(2*roc_auc - 1))

## 56, 55

# svm for all data
X_train = df1[df1['year']!=2017].drop('def_1', axis = 1).copy()
y_train = df1[df1['year']!=2017][['def_1']].copy()

X_test = df1[df1['year']==2017].drop('def_1', axis = 1).copy()
y_test = df1[df1['year']==2017][['def_1']].copy()

clf = svm.SVC(C=2, kernel = 'linear', probability=True, class_weight = 'balanced')
result=clf.fit(X_train, y_train['def_1'])
predicted = pd.DataFrame(result.predict_proba(X_train))[1]
roc_auc = roc_auc_score(y_true=y_train, y_score=predicted)
print('gini_tr = ', abs(2*roc_auc - 1))
predicted = pd.DataFrame(result.predict_proba(X_test))[1]
roc_auc = roc_auc_score(y_true=y_test, y_score=predicted)
print('gini_test = ', abs(2*roc_auc - 1))

## 60.4, 51



# вывод самого дерева в напечатанном виде
dt_plot(DecisionTree,X_train, X_test, y_train, y_test)


# Логит-модель с предсказаниями только в виде "0" и "1"
def logit_fun2(p):
    logit_model=sm.Logit(y_train,X_train[p])
    result=logit_model.fit()
    res = result.summary2()
    #print(res)
    predicted_tr = result.predict(X_train[p])
    roc_auc = roc_auc_score(y_true=y_train, y_score=predicted_tr)
    gini_tr = abs(2*roc_auc - 1)
    print('gini_train = ', gini_tr)
    predicted_ts = result.predict(X_test[p])
    roc_auc = roc_auc_score(y_true=y_test, y_score=predicted_ts)
    gini_ts = abs(2*roc_auc - 1)
    print('gini_test = ', gini_ts)
    
    predicted_tr = add_constant(predicted_tr)
    predicted_ts = add_constant(predicted_ts)
    
    dfcols_a = ['a', 'gini']
    A = pd.DataFrame(columns=dfcols_a)
    
    # Поиск наилучшей границы P*
    for a in list(predicted_tr[0].drop_duplicates()) + [0.99]:
        predicted_tr[1] = np.where( predicted_tr[0]>=a,1,0) #result.predict(predicted_tr)
        roc_auc = roc_auc_score(y_true=y_train, y_score=predicted_tr[1])
        gini_tr_new = abs(2*roc_auc - 1)
        #print('gini_train_new = ', gini_tr_new)
        A = A.append(pd.Series([a,gini_tr_new], index=dfcols_a), ignore_index=True)
    a = A[A['gini']==A['gini'].max()]['a'].mean()
    
    print("a = ", a)
    
    # Построение матриц кол-ва попаданий и ошибок первого и второго рода
    d = pd.DataFrame({'pred':predicted_tr[0],'fact':y_train['def_1']})
    d['pred1']=np.where(d['pred']>a,1,0)
    roc_auc = roc_auc_score(y_true=y_train, y_score=d['pred1'])
    gini_ts_new = abs(2*roc_auc - 1)
    print('gini_test_new = ', gini_ts_new)
    d['n']=1
    d1 = pd.DataFrame({'0_fact':[d[(d['pred1']==0)&(d['fact']==0)]['n'].sum(),d[(d['pred1']==1)&(d['fact']==0)]['n'].sum()],
                              '1_fact':[d[(d['pred1']==0)&(d['fact']==1)]['n'].sum(),d[(d['pred1']==1)&(d['fact']==1)]['n'].sum()]})
    print(d1)
    
    d = pd.DataFrame({'pred':predicted_ts[0],'fact':y_test['def_1']})
    d['pred1']=np.where(d['pred']>=a,1,0)
    roc_auc = roc_auc_score(y_true=y_test, y_score=d['pred1'])
    gini_ts_new = abs(2*roc_auc - 1)
    print('gini_test_new = ', gini_ts_new)
    d['n']=1
    d['n']=1
    d2 = pd.DataFrame({'0_fact':[d[(d['pred1']==0)&(d['fact']==0)]['n'].sum(),d[(d['pred1']==1)&(d['fact']==0)]['n'].sum()],
                              '1_fact':[d[(d['pred1']==0)&(d['fact']==1)]['n'].sum(),d[(d['pred1']==1)&(d['fact']==1)]['n'].sum()]})
    print(d2)
    
    return A, a, d1, d2

# Итоговое сравнение моделей
Model = 3.2

if Model==3.2:
    l = ';pr2;unpr1;num_all;fs6^1_2;pr4^1_2;re1^1_2;liq2^1_2;life;medical;2012;2013;const;'
else
    l = ';pr2;unpr1;num_all;fs6^1_2;pr4^1_2;re1^1_2;delta_prem^1_2;life;medical;2012;2013;const;'

l1 = l.split(';')
l1.remove('')
l1.remove('')

A, a, d1, d2 = logit_fun2(l1)

A = A.sort_values(by='a')

plt.figure()
lw = 2
plt.plot(A['a'], A['gini'], color='navy',
         lw=lw)
plt.xlim([0.0, 1.0])
plt.xlabel('P*')
plt.ylabel('Gini')
plt.savefig(ws+ "P_gini.png")
plt.show()


writer = pd.ExcelWriter(ws + 'd1.xlsx')
d1.to_excel(writer)
writer.save()

# Построение итоговой модели, обученной на всей выборке
logit_model=sm.Logit(df1['def_1'],df1[l1])
result=logit_model.fit()
res = result.summary2()
print(res)
predicted_tr = result.predict(df1[l1])
roc_auc = roc_auc_score(y_true=df1[['def_1']], y_score=predicted_tr)
gini = abs(2*roc_auc - 1)
print('gini_train = ', gini)

predicted_tr = add_constant(predicted_tr)

dfcols_a = ['a', 'gini']
A = pd.DataFrame(columns=dfcols_a)

# Поиск наилучшей границы P*
for a in list(predicted_tr[0].drop_duplicates()) + [0.99]:
    predicted_tr[1] = np.where( predicted_tr[0]>=a,1,0) #result.predict(predicted_tr)
    roc_auc = roc_auc_score(y_true=df1[['def_1']], y_score=predicted_tr[1])
    gini_tr_new = abs(2*roc_auc - 1)
    #print('gini_train_new = ', gini_tr_new)
    A = A.append(pd.Series([a,gini_tr_new], index=dfcols_a), ignore_index=True)
a = A[A['gini']==A['gini'].max()]['a'].mean()

print("a = ", a)

# Построение матриц кол-ва попаданий и ошибок первого и второго рода
d = pd.DataFrame({'pred':predicted_tr[0],'fact':y_train['def_1']})
d['pred1']=np.where(d['pred']>a,1,0)
roc_auc = roc_auc_score(y_true=df1[['def_1']], y_score=d['pred1'])
gini_ts_new = abs(2*roc_auc - 1)
print('gini_test_new = ', gini_ts_new)
d['n']=1
d1 = pd.DataFrame({'0_fact':[d[(d['pred1']==0)&(d['fact']==0)]['n'].sum(),d[(d['pred1']==1)&(d['fact']==0)]['n'].sum()],
                          '1_fact':[d[(d['pred1']==0)&(d['fact']==1)]['n'].sum(),d[(d['pred1']==1)&(d['fact']==1)]['n'].sum()]})
print(d1)
    
# Расчет VIF
X = add_constant(df1[l1])
vf = pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)

writer = pd.ExcelWriter(ws + 'vif.xlsx')
vf.to_excel(writer)
writer.save()


