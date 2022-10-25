#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Em-Fl (emilia.flo.rama at gmail)

"""
import pandas as pd
import numpy as np
import pickle
from os.path import join
import mne
from mne.channels import find_ch_adjacency
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import ptitprince as pt
from scipy.stats import zscore
from datetime import date

#========================================================================
#
# CLUSTER PERMUTATION ANALYSIS TIME DOMAIN
# independent groups with different number of observations
#
#========================================================================= 

def statistics_space_time(group_by,today,evoked_data_all, conditions,cfg,
                          overwrite):
    
    time_window = cfg.cl_stat_dict['time_window']
    n_permutations = cfg.cl_stat_dict['n_permutations']
    max_step = cfg.cl_stat_dict['max_step'] 
    p_threshold =  cfg.cl_stat_dict['init_p_value'] 
    n_conditions = len(conditions)
    n_observations = len(evoked_data_all[conditions[0]]) + len(evoked_data_all[conditions[1]])    
    dfn = n_conditions - 1  # degrees of freedom numerator
    dfd = n_observations - n_conditions  # degrees of freedom denominator
    thresh = scipy.stats.f.ppf(1 - p_threshold, dfn=dfn, dfd=dfd)  # F distribution
    tail =  cfg.cl_stat_dict['tail']                        
    
    cluster_name = 'HEP_' + group_by +'_' + today +'.cluster'

    cluster_path = join(cfg.statistics_path, cluster_name)
    
    
    X0 =  np.zeros(( len(evoked_data_all[conditions[0]]), len(evoked_data_all[conditions[0]][0].times),len(evoked_data_all[conditions[0]][0].info['ch_names']))) 
    X1 =  np.zeros(( len(evoked_data_all[conditions[1]]), len(evoked_data_all[conditions[1]][0].times),len(evoked_data_all[conditions[1]][0].info['ch_names']))) 
                     
    for idx,e in enumerate(evoked_data_all[conditions[0]]):
        data = e.data
        X0[idx,:,:] = data.T
        
    for idx,e in enumerate(evoked_data_all[conditions[1]]):
        data = e.data
        X1[idx,:,:] = data.T


    # adjacency, ch_names = read_ch_adjacency('biosemi64', picks=None) 
    adjacency, ch_names = find_ch_adjacency(evoked_data_all[conditions[0]][0].info,ch_type='eeg') 
    
    ## crop data on the time dimension
    times =  evoked_data_all[conditions[1]][0].times
    time_indices = np.logical_and(times >= time_window[0],
                                  times <= time_window[1])
 
    X0 = X0[:,time_indices,:]   
    X1 = X1[:,time_indices,:]   
    X = [X0,X1]            

    ## set up cluster analysis


    seed = 7 
    
    print('time window to analyse: ' + str(time_window[0]) +' to ' +str(time_window[1]))
    
    print('sample maxstep value is: ' + str(max_step))
    print('tails: ' + str(cfg.cl_stat_dict['tail']))
    print('sample p value is: ' + str(p_threshold))

   
    F_obs, clusters, cluster_p_values, H0 =  \
        mne.stats.permutation_cluster_test(X,
                                                  threshold=thresh,
                                                  n_permutations=n_permutations,
                                                  tail=tail,
                                                  seed=seed,
                                                  out_type='mask',
                                                  adjacency = adjacency,max_step=max_step,verbose='ERROR')

    if 'cluster_p_values' in locals():
        print(np.min(cluster_p_values))
                                         
        cluster_dict = dict(F_obs=F_obs, clusters=clusters,
                    cluster_p_values=cluster_p_values, H0=H0, tail=tail,time_window=time_window,
                    max_step=max_step, sample_p_value=p_threshold,n_permutations=n_permutations)

        
        sig_clusters = np.where(cluster_dict['cluster_p_values']<cfg.cluster_p_value)[0]
        if len(sig_clusters)!= 0:
            for cl in sig_clusters:
                electrodes = np.unique(np.where(cluster_dict['clusters'][cl])[1]) # get electrodes
                electrode_names = []
                for e in electrodes:
                    electrode_names.append(evoked_data_all[conditions[0]][0].info.ch_names[e])
            
                samples = np.unique(np.where(cluster_dict['clusters'][cl])[0]) # get time points
                print('cluster number: '+ str(cl))
                print('pvalue is: '+ str(cluster_dict['cluster_p_values'][cl]))
                print('elecs in cluster are\n',electrode_names)
                print('samples in cluster are\n', samples/int(evoked_data_all[conditions[0]][0].info['sfreq'])+time_window[0], '\n')
    
            fig, ax = plt.subplots(1,1)
            plt.imshow(cluster_dict['F_obs'].T,extent=[time_window[0],time_window[1],0,1], origin = 'lower',interpolation='none')
            plt.show()
            plt.pause(3)
        else:
            print('no significant cluster, minimum p cluster value is:  ' + str(np.min(cluster_p_values)))
        with open(cluster_path, 'wb') as filename:
            pickle.dump(cluster_dict, filename)

        print('finished saving cluster at path: ' + cluster_path)
        return cluster_dict
    
    else:
        print('no cluster found')

    
#========================================================================
#
# Stats on EEG MARKERS and plots
#========================================================================
def markers_stats_n_plot(group_by,markers_list,all_subj_markers,all_subj_epochs_markers,cfg):
    
    today = str(date.today())
    
    indep_var = group_by +'_category'

    path = join(cfg.markers_path,all_subj_markers)
    path_open = open(path , "rb")
    df_all_subj_markers = pickle.load(path_open) 

    path = join(cfg.markers_path,all_subj_epochs_markers)    
    path_open = open(path , "rb")
    df_all_subj_epochs_markers = pickle.load(path_open) 
    
    df_all_subj_markers['outcome_category'] =  df_all_subj_markers['outcome'].astype('category').cat.rename_categories({1.0: 'alive', 0.0: 'deceased'})   
    df_all_subj_markers['gose_dc_category'] = None
    df_all_subj_markers['command_score_dc_category'] = None    
    df_all_subj_markers['gose_dc_category']  = np.where(df_all_subj_markers['gose_dc'] > 3, 'good', 'bad' )
    df_all_subj_markers['gose_dc_category'] = df_all_subj_markers['gose_dc_category'].astype('category')   
    df_all_subj_markers['command_score_dc_category']  = np.where(df_all_subj_markers['command_score_dc'] > 3, 'good', 'bad' )
    df_all_subj_markers['command_score_dc_category'] = df_all_subj_markers['command_score_dc_category'].astype('category')
    
    df_all_subj_markers.to_pickle(join(cfg.markers_path ,'df_markers_all_subj_w_category.pkl'),protocol=4)  

    df_all_subj_epochs_markers['outcome_category'] =  df_all_subj_epochs_markers['outcome'].astype('category').cat.rename_categories({1.0: 'alive', 0.0: 'deceased'})   
    df_all_subj_epochs_markers['gose_dc_category'] = None
    df_all_subj_epochs_markers['command_score_dc_category'] = None  
    df_all_subj_epochs_markers['gose_dc_category']  = np.where(df_all_subj_epochs_markers['gose_dc'] > 3, 'good', 'bad' )
    df_all_subj_epochs_markers['gose_dc_category'] = df_all_subj_epochs_markers['gose_dc_category'].astype('category')   
    df_all_subj_epochs_markers['command_score_dc_category']  = np.where(df_all_subj_epochs_markers['command_score_dc'] > 3, 'good', 'bad' )
    df_all_subj_epochs_markers['command_score_dc_category'] = df_all_subj_epochs_markers['command_score_dc_category'].astype('category')
    
    df_all_subj_epochs_markers.to_pickle(join(cfg.markers_path ,'df_markers_all_subj_epochs_w_category.pkl'),protocol=4)   
        
    print('\n stats on good segments of eeg data')
    df_all_subj_markers = df_all_subj_markers[(df_all_subj_markers['eeg_segment_good']=='y') | (df_all_subj_markers['eeg_segment_good']=='yn')]      

    file_names = df_all_subj_markers['file_id']
    rename=[]
    for file in file_names:
        if file[:3]=='EBC':
            rename.append(file[3:6])
        else:
            rename.append(file[:3])
            
    df_all_subj_markers['id'] = rename    


    unique_id = np.unique(rename)
    df_all_subj_markers_select = pd.DataFrame(columns=list(df_all_subj_markers.columns.values))
    
    # keep first segment of data if subject has multiple good epochs
    for n in unique_id:
        row_y = list(np.where(df_all_subj_markers[df_all_subj_markers['id']==n]['eeg_segment_good']=='y')[0])
        if len(row_y)==0:
            row_y = list(np.where(df_all_subj_markers[df_all_subj_markers['id']==n]['eeg_segment_good']=='yn')[0])
            
        selected_files = df_all_subj_markers[df_all_subj_markers['id']==n]
        selected_files = selected_files.reset_index()
        selected_files.drop(labels=['index'],axis=1,inplace=True)
        selected_file = list(selected_files.loc[row_y[0]].values) 
        
        df_file = pd.DataFrame([selected_file],columns=list(df_all_subj_markers.columns.values))
        df_all_subj_markers_select = pd.concat([df_all_subj_markers_select,df_file])

    df_select =  df_all_subj_markers_select
    
    fig1, ax1 = plt.subplots(1,3)
    fig1.set_size_inches(12, 5)
    plt.subplots_adjust(left=0.2,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)


    stats_list = []
    pvalues = []
    
    power_marker_names = ['delta norm','theta norm', 'alpha norm', 'beta norm']
    other_marker_names = ['Kolmog complexity', 'permutation entropy', 'wSMI']
    
    for m,marker in enumerate(cfg.markers_list[0:3]):     # iterate over markers to get stats and plots
    
        df = df_select[[marker,indep_var]]
        
        groups = list(np.unique(df[indep_var]))
        
        if group_by!='outcome':
            groups.reverse()
            
        x = df[df[indep_var]==groups[0]][marker]
        y = df[df[indep_var]==groups[1]][marker]
        
        U1 , p = mannwhitneyu(x, y, use_continuity=True, alternative='two-sided')
        stats_list.append(U1)
        pvalues.append(p)
        
        
           
        pt.RainCloud(x = indep_var, y = marker, data = df, palette = cfg.palette, bw = .2,
                         width_viol = .6, ax = ax1[m], orient = 'v',pointplot = False,linecolor = 'grey',linewidth = 1,point_size = 3,order=groups)
        ax1[m].title.set_text(other_marker_names[m])
        ax1[m].set_xlabel('')
        ax1[m].set_ylabel('')
        # ADD TITLE OF GROUP 
    plt.suptitle(cfg.grouping_names[group_by])
    plt.show()
    fig1.savefig(join(cfg.images_path, group_by +'_other_markers_'+today+'.pdf'),bbox_inches='tight', dpi=300)
    plt.pause(3)
    plt.close('all')
    
    
    fig2, ax2 = plt.subplots(1,4)
    fig2.set_size_inches(12, 5)
    plt.subplots_adjust(left=0.2,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
        
    for m,marker in enumerate(cfg.markers_list[3:]):     # iterate over markers to get stats and plots
    
        df = df_select[[marker,indep_var]]
        
        groups = list(np.unique(df[indep_var]))
        
        if group_by!='outcome':
            groups.reverse()
            
        x = df[df[indep_var]==groups[0]][marker]
        y = df[df[indep_var]==groups[1]][marker]
       
        U1 , p = mannwhitneyu(x, y, use_continuity=True, alternative='two-sided')
        stats_list.append(U1)
        pvalues.append(p)
        
        
        
     
        pt.RainCloud(x = indep_var, y = marker, data = df, palette = cfg.palette, bw = .2,
                         width_viol = .6, ax = ax2[m], orient = 'v',pointplot = False,linecolor = 'grey',linewidth = 1,point_size = 3,order=groups)
        ax2[m].title.set_text(power_marker_names[m])
        ax2[m].set_xlabel('')
        ax2[m].set_ylabel('')
        
    plt.suptitle(cfg.grouping_names[group_by])
    plt.show()
    fig2.savefig(join(cfg.images_path, group_by+'_power_markers' +'_'+today+'.pdf'),bbox_inches='tight', dpi=300)
    plt.pause(3)
    plt.close('all')
            
  
    df_stats = pd.DataFrame({'markers':markers_list,'U':stats_list,'p':pvalues})
    print(df_stats)
    df_stats.to_pickle(join(cfg.statistics_path, 'markers_stats' + '_' + group_by +'_'+today+'.pkl'),protocol=4)

    return df_stats
     

#========================================================================
#
# Stats on ECG parameters (HR,HRV) and plots
#
#========================================================================

def ecg_param_stats_n_plot(df_file,group_by,cfg,overwrite):
    
    today = str(date.today())

    fig_name = 'patients_HR_HRV' + '_' + group_by +'_'+today+'.pdf'

    save_fig_path =join(cfg.images_path,fig_name)
             
    ecg_params= pd.read_pickle(df_file)
    ecg_params['outcome_category'] =  ecg_params['outcome'].astype('category').cat.rename_categories({1.0: 'alive', 0.0: 'deceased'})
    ecg_params['gose_dc_category'] = None
    ecg_params['command_score_dc_category'] = None   
    ecg_params['gose_dc_category']  = np.where(ecg_params['gose_dc'] > 3, 'good', 'bad' )
    ecg_params['gose_dc_category'] = ecg_params['gose_dc_category'].astype('category')   
    ecg_params['command_score_dc_category']  = np.where(ecg_params['command_score_dc'] > 3, 'good', 'bad' )
    ecg_params['command_score_dc_category'] = ecg_params['command_score_dc_category'].astype('category')
    
    ecg_params_w_category = ecg_params
    
    indep_var = group_by +'_category'
    dep_var = [ 'ECG_Rate_Mean','HRV_MeanNN']
    
    file_names = ecg_params['file_id']
    rename=[]
    for file in file_names:
        if file[:3]=='EBC':
            rename.append(file[3:6])
        else:
            rename.append(file[:3])
            
    ecg_params['id'] = rename 
    unique_id = np.unique(rename)
    ecg_params_select = pd.DataFrame(columns=list(ecg_params.columns.values))
    
    # keep first segment of data if subject has multiple good epochs
    for n in unique_id:
        row_y = list(np.where(ecg_params[ecg_params['id']==n]['ecg_segment_good']=='y')[0])
        if len(row_y)==0:
            row_y = list(np.where(ecg_params[ecg_params['id']==n]['ecg_segment_good']=='yn')[0])
            
        selected_files = ecg_params[ecg_params['id']==n]
        selected_files = selected_files.reset_index()
        selected_files.drop(labels=['index'],axis=1,inplace=True)
        selected_file = list(selected_files.loc[row_y[0]].values) 
        
        df_file = pd.DataFrame([selected_file],columns=list(ecg_params.columns.values))
        ecg_params_select = pd.concat([ecg_params_select,df_file])

                    
    fig1, ax1 = plt.subplots(2,1)
    fig1.set_size_inches(7, 5)
    plt.subplots_adjust(left=0.2,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
    groups = list(np.unique(ecg_params[indep_var]))

    if group_by!='outcome':
        groups.reverse()

    pt.RainCloud(x = indep_var, y = dep_var[0], data = ecg_params_select, palette = cfg.palette, bw = .2,
                     width_viol = .6, ax = ax1[0], orient = 'h',pointplot = False,linecolor = 'grey',linewidth = 1,point_size = 3,order=groups)

    ax1[0].set_ylabel('')
    ax1[0].set_xlabel('Heart rate',fontsize=14)
    ax1[0].tick_params(axis = 'both', which = 'major', labelsize = 14)
        
    pt.RainCloud(x = indep_var, y = dep_var[1], data = ecg_params_select, palette =cfg.palette, bw = .2,
                     width_viol = .6, ax = ax1[1], orient = 'h',pointplot = False,linecolor = 'grey',linewidth = 1,point_size = 3,order=groups)

    ax1[1].set_ylabel('')
    ax1[1].set_xlabel('Heart rate variability',fontsize=14)
    ax1[1].tick_params(axis = 'both', which = 'major', labelsize = 14)
    plt.suptitle(group_by)

    plt.show()
    plt.pause(2)
    fig1.savefig(save_fig_path,bbox_inches='tight', dpi=300)
    plt.close()  
    
    # STATS #
    
    stats_list = []
    pvalues = []
    for var in dep_var:
        df = ecg_params_select[[var,indep_var]]
    
        groups = list(np.unique(df[indep_var]))
        if indep_var!='outcome':
            groups.reverse()
        
        print(groups)
    
        x = df[df[indep_var]==groups[0]][var]
        y = df[df[indep_var]==groups[1]][var]
   
        U1 , p = mannwhitneyu(x, y, use_continuity=True, alternative='two-sided')
        stats_list.append(U1)
        pvalues.append(p)
        
    df_stats = pd.DataFrame({'dep variable':dep_var,'U':stats_list,'p':pvalues})
    print('\n stats for ' + group_by +'\n')
    print(df_stats)
    df_stats.to_pickle(join(cfg.statistics_path, 'HR_HRV' + '_' + group_by +'_'+today+'.pkl'),protocol=4)
    
    ecg_params_w_category.to_pickle(join(cfg.ecg_param_path ,'ecg_all_subj_w_category.pkl'),protocol=4)  

    return df_stats

#========================================================================
#
# Correlation between markers and HEP amplitude
#
#========================================================================

def HEP_markers_corr(df_summary,group_by,cfg):
    
    group = group_by +'_category'    
    
    df_all = pd.read_pickle(join(cfg.summary_path,df_summary))
    df_all.dropna(inplace=True)
    df_all.reset_index(inplace=True)
    df_all = df_all[(df_all['HEP_data_good']=='y') | (df_all['HEP_data_good']=='yn')]
    
    
    file_names = df_all['file_id']
    rename=[]
    for file in file_names:
        if file[:3]=='EBC':
            rename.append(file[3:6])
        else:
            rename.append(file[:3])
            
    df_all['id'] = rename 
    unique_id = np.unique(rename)
    segments_select = pd.DataFrame(columns=list(df_all.columns.values))
    
    # keep first segment of data if subject has multiple good epochs
    for n in unique_id:
        row_y = list(np.where(df_all[df_all['id']==n]['HEP_data_good']=='y')[0])
        if len(row_y)==0:
            row_y = list(np.where(df_all[df_all['id']==n]['HEP_data_good']=='yn')[0])
            
        selected_files = df_all[df_all['id']==n]
        selected_files = selected_files.reset_index()
        selected_files.drop(labels=['index'],axis=1,inplace=True)
        selected_file = list(selected_files.loc[row_y[0]].values) 
        
        df_file = pd.DataFrame([selected_file],columns=list(df_all.columns.values))
        segments_select = pd.concat([segments_select,df_file])
        
    df = segments_select
    
    df_200_400_frontal = df.filter(regex='F3_200_450|F4_200_450|F7_200_450|F8_200_450|Fz_200_450|Fp1_200_450|Fp2_200_450|Fpz_200_450|file_id|file_id_segment|HEP_data_good|eeg_segment_good|ecg_segment_good')
    df_200_400_central_posterior = df.filter(regex='C3_200_450|C4_200_450|O1_200_450|O2_200_450|Cz_200_450|P3_200_450|P4_200_450|Pz_200_450|file_id|file_id_segment|HEP_data_good|eeg_segment_good|ecg_segment_good')
    
    df_200_400_frontal_mean = np.mean(df_200_400_frontal.filter(regex='200_450'),axis=1)
    df_200_400_central_posterior_mean = np.mean(df_200_400_central_posterior.filter(regex='200_450'),axis=1)
    
    df_markers = df[['file_id','kolcom', 'p_e', 'wSMI', 
                    'delta', 'delta_n', 'theta', 'theta_n', 
                    'alpha', 'alpha_n', 'beta', 'beta_n','command_score_dc_category','gose_dc_category','outcome' ]]
    
    df_markers_voltage = df_markers
    df_markers_voltage['frontal_HEP'] = df_200_400_frontal_mean
    df_markers_voltage['central_posterior_HEP'] = df_200_400_central_posterior_mean

    for marker in cfg.markers_list:
        
        save_fig_path = join(cfg.images_path,'HEP_marker_corr_'+group+'_'+marker+'.pdf')
        
        df_markers_voltage["zscore"] = zscore(df_markers_voltage[marker])
        df_markers_voltage["is_outlier"] =  df_markers_voltage["zscore"].apply(lambda x: x <= -3 or x >= 3)
        df_no_outliers = df_markers_voltage[df_markers_voltage["is_outlier"]==False]
    
        fig1, ax1 = plt.subplots(1,2)
        fig1.set_size_inches(10, 5)
        plt.subplots_adjust(left=0.1,
                            bottom=0.2, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.3, 
                            hspace=0.2)
        
        sns.scatterplot(data=df_no_outliers, x="frontal_HEP", y=marker, hue=group,ax=ax1[0],palette=cfg.palette)
        ax1[0].set_ylabel(cfg.marker_names[marker],fontsize=12)
        ax1[0].set_xlabel('mean voltage',fontsize=12)
        ax1[0].tick_params(axis = 'both', which = 'major', labelsize = 12)
        ax1[0].title.set_text('frontal ch')
        ax1[0].legend().set_visible(False)
        
        
        sns.scatterplot(data=df_no_outliers, x="central_posterior_HEP", y=marker, hue=group, ax=ax1[1],palette=cfg.palette)
        ax1[1].set_ylabel(cfg.marker_names[marker],fontsize=12)
        ax1[1].set_xlabel('mean voltage',fontsize=12)
        ax1[1].tick_params(axis = 'both', which = 'major', labelsize = 12)
        ax1[1].title.set_text('central posterior ch')
        ax1[1].legend(loc='upper center', bbox_to_anchor=(-0.23, -0.08),
          fancybox=False, shadow=True, ncol=2,title=cfg.grouping_names[group_by])
        
        s,p = stats.spearmanr(df_no_outliers[marker], df_no_outliers['frontal_HEP'])
        print('for frontal HEP and ', marker,':')
        print('Spearmean s corr: ',str(s))
        print('p value for a two tailed t test: ',str(p))
        
        s,p = stats.spearmanr(df_no_outliers[marker], df_no_outliers['central_posterior_HEP'])
        print('for central_posterior HEP and ', marker,':')
        print('Spearmean s corr: ',str(s))
        print('p value for a two tailed t test: ',str(p))       
        
        plt.show()
        plt.pause(2)
        
        fig1.savefig(save_fig_path,bbox_inches='tight', dpi=300)
        plt.close()  
  
   