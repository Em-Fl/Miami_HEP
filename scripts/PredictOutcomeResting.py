# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from os.path import join
import csv
from datetime import datetime
import pickle as pkl

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import roc_curve, roc_auc_score, r2_score
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import RidgeCV,Ridge

from imblearn.over_sampling import SMOTE

from scipy.stats import ttest_ind, tukey_hsd, norm
import pymrmr
from itertools import chain

def getAnonName(filename):
    ind = filename.index('_')
    name = filename[0:ind].replace('~','').strip()
    return name

def binarizeClassLabel(y,thresh):
    return [0 if x < thresh else 1 for x in y]

def convertToCSV(row):
    outcomerows = []
    for r in row:
        tmp = ''
        for y in r:
            # mean + 95% CI
            mu = np.mean(y)
            conf = norm.interval(0.95, loc=mu, scale=np.std(y)/np.sqrt(len(y)))
            tmp = tmp + ('{:4.2f}'.format(mu) + ' [ ' + '{:4.2f}'.format(conf[0]) + ' ' + '{:4.2f}'.format(conf[1]) + '],')
        outcomerows.append(tmp[:-1])
    return outcomerows
  
        
RS = 63 #np.random.RandomState(0) #42 #63 # random state
N_JOBS = -1  # number of parallel threads, -1 for all available cores

N_SPLITS = 10
N_REPEATS = 100
N_REPEATS2 = 1000
do_feat_select = False
method = 'mrmr'
do_trial_avg = False

# ### Read Data from Excel Files
base_path = os.getcwd()
dpath = join(base_path,'HR_HEP_and_markers_2023-08-28.csv')
dpath2 = join(base_path,'Resting State Revisions.xlsx')
if os.path.exists(dpath):
    is_directory = os.path.isdir(dpath)
    is_file = os.path.isfile(dpath)

    if is_directory:
        print('Path points to a directory. Reading the following files in ', dpath)
        file_list = [join(dpath, f) for f in os.listdir(dpath) if (os.path.isfile(join(dpath, f)) & f.endswith(".csv"))]
        print(file_list)
        file_name = dpath.replace('/','_') + '.csv'
    elif is_file:
        print('Path points to a file. Reading the following file in ', os.path.dirname(dpath))
        file_list = [dpath]
        print(file_list)
        file_name = os.path.basename(dpath)

    print(file_name)
    li = []
    for f in file_list:
        print(f)
        df = pd.read_csv(f)
        li.append(df)
    df = pd.concat(li, sort=False)
    df.reset_index(drop=True, inplace=True)
    print('Train data size: ', df.shape)
else:
    print('Path does not exist! check dpath variable.')



idx2class_all = [{0: 'dead', 1: 'alive'},
             {0: '< MCS+', 1: '>= MCS+'},
             {0: 'low', 1: 'high'}]

# binarize class labels
df.gose_dc = binarizeClassLabel(df.gose_dc.values.tolist(), 4)
df.command_score_dc = binarizeClassLabel(df.command_score_dc.values.tolist(), 4)

# create anonymized names column
df['anonname'] = [getAnonName(x) for x in df['file_id'].values.tolist()]


# ### Create Input and Output Data
complexity = ['kolcom']

pe = ['p_e']

spec = [
'delta_n',
'theta_n',
'alpha_n',
'beta_n']

wSMI = ['wSMI']

# mEmCh and mEsdCh columns
nonspeceeg = df.columns[72:75].tolist()+df.columns[83:86].tolist()
speceeg = df.columns[75:82][::2].tolist()+df.columns[86:93][::2].tolist()
eeg_cols = nonspeceeg+speceeg

outcome_cols = [
    'outcome',
    'gose_dc',
    'command_score_dc']

clin_cols = [
    'Age',
    'L_reactivity',
    'R_reactivity',
    'Motor_GCS_admission']

ct_col = ['Marshall_ct_classification']

hep_cols = ['Cz_0_600','Cz_600_800']

cols_list = [[clin_cols,
              ct_col,
              eeg_cols,
              clin_cols + ct_col,
              clin_cols+ct_col+eeg_cols,
              clin_cols+eeg_cols,
              ct_col+eeg_cols],
            [clin_cols,
             ct_col,
             hep_cols,
              clin_cols + ct_col,
              clin_cols+ct_col+hep_cols,
              clin_cols+hep_cols,
              ct_col+hep_cols,
              clin_cols+ct_col+eeg_cols+hep_cols,
              clin_cols+ct_col+eeg_cols,
              hep_cols+eeg_cols,
              eeg_cols]]

ncombs = len(cols_list[0]) + len(cols_list[1])
test_pairs = [(3,0),(4,3),(5,0),(6,1),(10,7),(11,10),(12,7),(13,8),(14,10),(14,15),(16,9),(2,5),(9,17)]
nmodels = 4

# prepare datasets
df_eeg = df.dropna(subset = clin_cols+ct_col+eeg_cols+outcome_cols)
df_hep = df.dropna(subset = clin_cols+ct_col+eeg_cols+hep_cols+outcome_cols)
if do_trial_avg:
    df_eeg = df_eeg.groupby('anonname').mean()
    df_hep = df_hep.groupby('anonname').mean()
else:
    
    # Emilia's annotations
    df_eeg = df_eeg.loc[df_eeg['in_group_analysis_markers'] == 1]
    df_hep = df_hep.loc[df_hep['in_group_analysis_markers'] == 1]
    

dfs = [df_eeg, df_hep]
#results: 4 models, 3 outcomes, 18 datasets
r2_all = []
tprs_all = []
aucs_all = []
precs_all = []
recs_all = []
fs_all = []
fs_all2 = []

mean_fpr = np.linspace(0,1,N_REPEATS)
states = np.random.RandomState(0).randint(2**16,size=N_REPEATS)
states2 = np.random.RandomState(0).randint(2**16,size=N_REPEATS2)
auc_dists = np.zeros([ncombs,len(outcome_cols),nmodels,N_REPEATS])
f_dists = np.zeros([ncombs,len(outcome_cols),nmodels,N_REPEATS])
f_dists2 = np.zeros([ncombs,len(outcome_cols),nmodels,N_REPEATS])
i = 0
for c in range(len(cols_list)):
    print('Using dataset: ' + str(c))
    ds = dfs[c]
    cols_list2 = cols_list[c]
    for ind in range(len(cols_list2)):
        selected_cols = cols_list2[ind]
        print('Using features: ' + str(selected_cols))
        # Outcome, Gose, Command Score
        tprs = []
        aucs = []
        precs = []
        recs = []
        fs = []
        fs2 = []
        for x in range(len(outcome_cols)):
            tprs_tmp = [[] for x in range(nmodels)]
            aucs_tmp = [[] for x in range(nmodels)]
            prec_tmp = [[] for x in range(nmodels)]
            rec_tmp = [[] for x in range(nmodels)]
            f_tmp = [[] for x in range(nmodels)]
            f_tmp2 = [[] for x in range(nmodels)]
            fi_tmp = []
            idx2class = idx2class_all[x]
            outcome = outcome_cols[x]
            print('Outcome: ' + outcome)
            df_clean = ds[selected_cols+[outcome]]
            for j in range(N_REPEATS):

                RS = states[j]
     
                # R-squared for all features (only needed once per assessment)
                if j == 0:
                    xr = df_clean[selected_cols+[outcome]]
                    corr_matrix = xr.corr()
                    rvec = corr_matrix[outcome].values.tolist()[:-1]
                    r2 = [n**2 for n in rvec]
                    r2_all.append(r2)
                feature_cols = selected_cols
                X = df_clean[feature_cols]
                y = df_clean[outcome]   
       
                # ### SVM with balanced dataset
                smote_obj = SMOTE(k_neighbors=1, random_state=RS)
                
                
                clf = svm.SVC(random_state=RS,kernel='linear',max_iter=100000)
                clf2 = RandomForestClassifier(random_state=RS, n_jobs=N_JOBS,n_estimators=500)
                clf3 = HistGradientBoostingClassifier(random_state=RS)
                clf4 = XGBClassifier(seed=RS, n_jobs=N_JOBS)
                cv = StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=RS)
                
                for train,test in cv.split(X,y):
                    df_tmp = pd.DataFrame()
                    df_tmp['outcome'] = y.iloc[train]
                    # only for EEG features, to battle curse of dimensionality
                    if do_feat_select and len(feature_cols) > 8:
                        # discretize train set
                        if method == 'mrmr':
                            for feat in selected_cols:
                                f = np.array(X[feat].iloc[train])
                                m = np.mean(f)
                                s = np.std(f)
                                bins = [-np.Inf,m-s,m+s,np.Inf]
                                x_dig = np.digitize(f,bins)
                                df_tmp[feat] = x_dig
                            # use 60% of all features
                            #nf = np.floor(len(selected_cols)*0.6)
                            # use 10% of training size
                            nf = np.floor(len(train)*0.1)
                            good_cols = pymrmr.mRMR(df_tmp,'MID',nf)
                            Xd = X[good_cols]
                        elif method == 'rfe':
                            estimator = svm.SVC(kernel='linear')
                            selector = RFECV(estimator, step=1, cv=5, n_jobs = N_JOBS)
                            selector = selector.fit(X,y)
                            mask = selector.get_support()
                            Xd = X.loc[:,mask]   
                    else:
                        Xd = X
                            
                    scaler = StandardScaler().fit(Xd.iloc[train])
                    X_train = np.array(scaler.transform(Xd.iloc[train]))
                    y_train = np.array(y.iloc[train])
                    X_test = np.array(scaler.transform(Xd.iloc[test]))
                    y_test = np.array(y.iloc[test])
                    X_train, y_train = smote_obj.fit_resample(X_train,y_train)
                    
    
                    pred = clf.fit(X_train,y_train).predict(X_test)
                    fpr, tpr, _ = roc_curve(y_test,pred)
                    tprs_tmp[0].append(np.interp(mean_fpr,fpr,tpr))
                    a = roc_auc_score(y_test,pred)
                    aucs_tmp[0].append(a)
                    auc_dists[i][x][0][j] = a
                    # precision and recall
                    [prec,rec,f,_] = precision_recall_fscore_support(y_test,pred,zero_division=np.nan)
                    prec_tmp[0].append(prec)
                    rec_tmp[0].append(rec)
                    f_tmp[0].append(f[0])
                    f_tmp2[0].append(f[1])
                    f_dists[i][x][0][j] = f[0]
                    f_dists2[i][x][0][j] = f[1]
                    #fi_tmp.append(clf.coef_[0])
                    
                    if nmodels > 1:
                        pred = clf2.fit(X_train,y_train).predict(X_test)
                        fpr, tpr, _ = roc_curve(y_test,pred)
                        tprs_tmp[1].append(np.interp(mean_fpr,fpr,tpr))
                        a = roc_auc_score(y_test,pred)
                        aucs_tmp[1].append(a)
                        auc_dists[i][x][1][j] = a
                        # precision and recall
                        [prec,rec,f,_] = precision_recall_fscore_support(y_test,pred,zero_division=np.nan)
                        prec_tmp[1].append(prec)
                        rec_tmp[1].append(rec)
                        f_tmp[1].append(f[0])
                        f_tmp2[1].append(f[1])
                        f_dists[i][x][1][j] = f[0]
                        f_dists2[i][x][1][j] = f[1]
                    
                    if nmodels > 2:
                        pred = clf3.fit(X_train,y_train).predict(X_test)
                        fpr, tpr, _ = roc_curve(y_test,pred)
                        tprs_tmp[2].append(np.interp(mean_fpr,fpr,tpr))
                        a = roc_auc_score(y_test,pred)
                        aucs_tmp[2].append(a)
                        auc_dists[i][x][2][j] = a
                        # precision and recall
                        [prec,rec,f,_] = precision_recall_fscore_support(y_test,pred,zero_division=np.nan)
                        prec_tmp[2].append(prec)
                        rec_tmp[2].append(rec)
                        f_tmp[2].append(f[0])
                        f_tmp2[2].append(f[1])
                        f_dists[i][x][2][j] = f[0]
                        f_dists2[i][x][2][j] = f[1]
                    
                    if nmodels > 3:
                        pred = clf4.fit(X_train,y_train).predict(X_test)
                        fpr, tpr, _ = roc_curve(y_test,pred)
                        tprs_tmp[3].append(np.interp(mean_fpr,fpr,tpr))
                        a = roc_auc_score(y_test,pred)
                        aucs_tmp[3].append(a)
                        auc_dists[i][x][3][j] = a
                        # precision and recall
                        [prec,rec,f,_] = precision_recall_fscore_support(y_test,pred,zero_division=np.nan)
                        prec_tmp[3].append(prec)
                        rec_tmp[3].append(rec)
                        f_tmp[3].append(f[0])
                        f_tmp2[3].append(f[1])
                        f_dists[i][x][3][j] = f[0]
                        f_dists2[i][x][3][j] = f[1]
    
            tprs.append([[np.mean(np.array(x),axis=0),np.std(np.array(x),axis=0)] for x in tprs_tmp])
            aucs.append(aucs_tmp)
            precs.append([[[np.nanmean(np.array(x[m])),np.nanstd(np.array(x[m]))] for m in range(2)] for x in prec_tmp])
            recs.append([[[np.nanmean(np.array(x[m])),np.nanstd(np.array(x[m]))] for m in range(2)] for x in rec_tmp])
            fs.append([[[np.nanmean(np.array(x[m])),np.nanstd(np.array(x[m]))] for m in range(2)] for x in f_tmp])
            fs2.append([[[np.nanmean(np.array(x[m])),np.nanstd(np.array(x[m]))] for m in range(2)] for x in f_tmp2])
        tprs_all.append(tprs)
        aucs_all.append(aucs)
        precs_all.append(precs)
        recs_all.append(recs)
        fs_all.append(fs)
        fs_all2.append(fs2)
        i += 1
    

# conduct one-tailed hypothesis tests on model improvement from feature addition
ps_all = np.zeros([len(test_pairs),len(outcome_cols),nmodels])
ts_all = np.zeros([len(test_pairs),len(outcome_cols),nmodels])
ps_all_f_neg = np.zeros([len(test_pairs),len(outcome_cols),nmodels])
ts_all_f_neg = np.zeros([len(test_pairs),len(outcome_cols),nmodels])
ps_all_f_pos = np.zeros([len(test_pairs),len(outcome_cols),nmodels])
ts_all_f_pos = np.zeros([len(test_pairs),len(outcome_cols),nmodels])
for i in range(len(outcome_cols)):
    for j in range(nmodels):
        for k in range(len(test_pairs)):
            pair = test_pairs[k]
            # auc tests
            ttr = ttest_ind(aucs_all[pair[0]][i][j],aucs_all[pair[1]][i][j],alternative='greater')
            ps_all[k][i][j] = ttr.pvalue
            ts_all[k][i][j] = ttr.statistic
            # f-measure tests, negative class
            ttrf = ttest_ind(f_dists[pair[0]][i][j],f_dists[pair[1]][i][j],alternative='greater')
            ps_all_f_neg[k][i][j] = ttrf.pvalue
            ts_all_f_neg[k][i][j] = ttrf.statistic
            # f-measure tests, positive class
            ttrf = ttest_ind(f_dists2[pair[0]][i][j],f_dists2[pair[1]][i][j],alternative='greater')
            ps_all_f_pos[k][i][j] = ttrf.pvalue
            ts_all_f_pos[k][i][j] = ttrf.statistic
        

# conduct Tukey's Range Test on distributions of AUCs for each model + feature set + outcome
anovas_p = np.zeros([ncombs,nmodels-1])
anovas_f = np.zeros([ncombs,nmodels-1])
total = ncombs*(nmodels-1)
svm_p_count = 0
for i in range(ncombs):
    dists = aucs_all[i][2]
    res = tukey_hsd(dists[0],dists[1],dists[2],dists[3])
    stats = res.statistic[0][1:]
    pvalues = res.pvalue[0][1:]
    for k in range(nmodels-1):
        anovas_f[i][k] = stats[k]
        anovas_p[i][k] = pvalues[k]
        if pvalues[k] < 0.05:
            svm_p_count += 1
print(svm_p_count/total)

dt = datetime.today().strftime('%Y-%m-%d')
# save variables
f = open(join(base_path,'Results\\rest_vars_' + dt + '.pckl'), 'wb')
pkl.dump([aucs_all,tprs_all,r2_all,ps_all,ts_all,ps_all_f_pos,ts_all_f_pos,ps_all_f_neg,ts_all_f_neg,fs_all,precs_all,recs_all,tprs_all],f)
f.close()


# Load pkl (COMMENT OUT IF NOT USING)
# f = open(join(base_path,'Results\\rest_vars_2024-01-10.pckl'),'rb')
# [aucs_all,tprs_all,r2_all,ps_all,ts_all,ps_all_f_pos,ts_all_f_pos,ps_all_f_neg,ts_all_f_neg,fs_all,precs_all,recs_all,tprs_all] = pkl.load(f)
# f.close()


with open(join(base_path,'Results\\results_auc_' + dt+ '.csv'),'w',newline='') as f:
    csvwriter = csv.writer(f, delimiter=',')
    csvwriter.writerow(['SVM','Random Forest','HistGBC','XGB'])
    for row in aucs_all:
        for r in convertToCSV(row):
            csvwriter.writerow(r.split(','))
            

with open(join(base_path,'Results\\results_r2_' + dt+ '.csv'),'w',newline='') as f:
    csvwriter = csv.writer(f, delimiter=',')
    csvwriter.writerow(clin_cols+ct_col+eeg_cols+hep_cols)
    for row in r2_all:
        csvwriter.writerow(row)

with open(join(base_path,'Results\\results_tpr_' + dt+ '.csv'),'w',newline='') as f:
    csvwriter = csv.writer(f, delimiter=',')
    for ds in tprs_all:
        for o in ds:
            for x in range(4):
                m = o[x]
                for i in m:
                    csvwriter.writerow(i)
    
with open(join(base_path,'Results\\results_p_' + dt+ '.csv'),'w',newline='') as f:
    csvwriter = csv.writer(f, delimiter=',')
    csvwriter.writerow(['SVM','','Random Forest','','HistGBC','','XGB'])
    for r in range(len(ps_all)):
        pr = ps_all[r]
        tr = ts_all[r]
        for i in range(3):
            csvwriter.writerow(chain.from_iterable(zip(tr[i],pr[i])))
            
            
            
# predict time to recovery for patients who recovered consciousness on discharge (command_score_dc) using EEG features only
# select relevant patients
f = open(dpath2,'rb')
df_ttr = pd.read_excel(f)
f.close()
df_ttr = df_ttr[df_ttr.Date_Recovery.notnull()]
# build outcomes by finding timedelta for patients who recovered
ttrs = ((df_ttr['Date_Recovery']-df_ttr['Date_EEG'])/pd.Timedelta(days=1)).values.tolist()
ttrs = [max(0.0,x) for x in ttrs]
# normalize ttrs
ttrs = [(x-min(ttrs))/max(ttrs) for x in ttrs]
subs_ttr = df_ttr['ID'].astype(str) 
outcomes = {}
for i in range(len(subs_ttr)):
    sub = subs_ttr.values.tolist()[i]
    ttr = ttrs[i]
    # remove nans
    if ttr != ttr:
        continue
    outcomes[sub] = ttr
    
# connect EEG dataset to TTR using patient ID
inds = df_eeg['patient_id'].isin(subs_ttr)
df_eeg_ttr = df_eeg[inds]
subs_eeg = df_eeg_ttr['patient_id'].values.tolist()
out_ttr = []
for sub in subs_eeg:
    if sub in outcomes:
        out_ttr.append(outcomes[sub])
outcome = 'TTR'
df_eeg_ttr.insert(0,outcome,out_ttr)
ttr_scores = np.zeros(N_REPEATS2)
ttr_scores_na = np.zeros(N_REPEATS2)
r2_all_ttr = []
ridges = []
for j in range(N_REPEATS2):

    RS = states2[j]
    
    # R-squared for all features (only needed once per assessment)
    if j == 0:
        xr = df_eeg_ttr[eeg_cols+clin_cols+[outcome]]
        corr_matrix = xr.corr()
        rvec = corr_matrix[outcome].values.tolist()[:-1]
        r2 = [n**2 for n in rvec]
        r2_all_ttr.append(r2)
    feature_cols = eeg_cols
    X = df_eeg_ttr[feature_cols]
    y = df_eeg_ttr[outcome]   
    
    cv = KFold(n_splits=N_SPLITS,shuffle=True,random_state=RS)
    tmpscores = []
    tmpsna = []
    for train,test in cv.split(X,y):
        # scale training set
        scaler = StandardScaler().fit(X.iloc[train])
        X_train = np.array(scaler.transform(X.iloc[train]))
        y_train = np.array(y.iloc[train])
        X_test = np.array(scaler.transform(X.iloc[test]))
        y_test = np.array(y.iloc[test])
        
        # RFE feature selection to fix dimensionality
        estimator = svm.SVR(kernel='linear')
        selector = RFE(estimator, step=1, n_features_to_select=3)
        selector = selector.fit(X_train,y_train)
        mask = selector.get_support()
        X_train = X_train[:,mask]
        X_test = X_test[:,mask]
        
        # use CV to select alpha
        rcv = RidgeCV(alphas=[0.001,0.01,0.1,1.0,10.0,100.0],gcv_mode='svd',cv=KFold(10))
        rcv.fit(X_train,y_train)
        ridges.append(rcv.alpha_)
        
        # ridge regression
        clf = Ridge(alpha=rcv.alpha_,solver='svd')
        pred = clf.fit(X_train,y_train).predict(X_test)
        score = r2_score(y_test,pred)
        tmpsna.append(score)
        # adjusted R-squared
        adjscore = 1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1])
        tmpscores.append(adjscore)
    ttr_scores[j] = np.mean(tmpscores)
    ttr_scores_na[j] = np.mean(tmpsna)
   
r_dict = {}
for r in ridges:
    if r not in r_dict:
        r_dict[r] = 0
    r_dict[r] += 1