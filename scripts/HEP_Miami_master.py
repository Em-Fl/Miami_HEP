#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Em-Fl (emilia.flo.rama at gmail)

"""
# at icm, environment clone_mne 
# matplotlib.use 'Qt5Agg'
# mne 24


#%% IMPORT PYTHON LIBRARIES

import matplotlib 
matplotlib.use('Qt5Agg')# To select bad hearbeats iteratively
import pandas as pd
import os
from os import chdir
import glob
import mne
mne.viz.set_browser_backend('matplotlib')  
from datetime import date
import matplotlib.pyplot as plt

plt.close('all')
# %% IMPORT FUNCTIONS SPECIFIC TO THIS PROJECT

# CHANGE PATH ACCORDINGLY !
# project_path ='/DATA1/Dropbox/PhD/Project_HEP/Miami_HEP'
project_path ='/home/emilia.ramaflo/Dropbox/PhD/Project_HEP/Miami_HEP'

script_path  = glob.glob(project_path +'/**/HEP_Miami_master.py', recursive=True)[0]
all_script_path = os.path.dirname(script_path)

chdir(all_script_path)

from preproc_HEP_Miami import create_patient_info, filter_raw_hp_lp, find_R_Peaks_4_good_segments,epoch_data_n_clean_autoreject
from evoked_HEP_Miami import get_evokeds, grand_average_evokeds,get_average_voltage_HEP
from plot_HEP_Miami import plot_compare_evokeds,plot_compare_topography,plot_significant_electrodes
from ECG_HEP_Miami import get_ecg_parameters
from stats_HEP_Miami import statistics_space_time, markers_stats_n_plot,ecg_param_stats_n_plot,HEP_markers_corr
from doc_markers_HEP_Miami import get_eeg_markers
from summary_HEP_Miami import final_summary_df
import configs_HEP_Miami as cfg # all parameters are here

#%%
###############################################################################
#                             GET PATIENT FILES
###############################################################################

chdir(cfg.raw_path)
all_files = glob.glob('*.edf')  # get all edfs  
all_files.sort()
files_id = [file[:12] for file in all_files]

files_no_good_segments = ['2_10a2e7bb-3','219_e4c9f429','230_6b99aca5',  # file_id of files without 10 minutes of good data (output from find R peaks 4 good segments)
                          '284_0335e2c8','338_80a985a3','450_8ed00e57',
                          '462_35fca0ca','528_023f9fec','546_ded15431',
                          '628_d42efbfd','712_64f2adf2','743_b0963592',
                          'EBC013_c0766', 'EBC030_c9971','EBC034_f8498',
                          'EBC045_ede7c']
files_no_ecg = ['EBC012_270fe','EBC012_660f3','EBC012_e772d', # file with no ecg signal in ecg channels 
                'EBC012_eacec','EBC012_eeeee','EBC012_f593b',
                'EBC013_c9b6a','EBC013_e4b40','EBC025_884af',
                'EBC035_69e7a','EBC045_5c8c7','EBC045_90af4',
                'EBC030_5c63e']

[files_id.remove(f) for f in files_no_good_segments]
[files_id.remove(f) for f in files_no_ecg]

ids = [file.split('_')[0] for file in files_id]
ids = [*set(ids)]
ids.sort()



# df init to fill with all subj data
ecg_all_subj = pd.DataFrame(data=cfg.df_ecg)
df_epochs_all_subj = pd.DataFrame(columns = cfg.column_names_markers)
df_all_subj = pd.DataFrame(columns = cfg.column_names_markers)
mean_HEP_all_subj = pd.DataFrame(columns = cfg.columns_HEP)

###############################################################################
#                            SELECT GROUPPING FACTOR AND FILES
###############################################################################

# WHAT TO GROUP: outcome, command_score_dc, gose_dc ##

group_by ='command_score_dc'  
epoch_type = 'HEP'  #HEP or marker

# FILES TO PREPROCESS
files_to_run = (0,None) 

overwrite = True # overwrite files for functions with overwrite parameter
plot_R_detection = False # plot R peak detection in 'find_R_Peaks_4_good_segments'
visual_check = True # plot overlapping Heartbeats to remove manually wrongly detected peaks in 'find_R_Peaks_4_good_segments'
plot = True # plot evokeds
manual_ch_rejection = True # second run of 'epoch_data_n_clean_autoreject', plots evoked for HEP and allows to select channels to interpolate that were not picked up by autoreject
#%%


###############################################################################
#                            SELECT OPERATIONS TO APPLY TO FILES
###############################################################################
operations_to_apply = dict(

                    ## Go!                                      
                    create_patient_info = 0,# info about state, outcome, determines montage,finds flat segments of eeg
                    filter_raw_hp_lp = 0, # filter data and select ecg channel subtraction
                    
                    find_R_Peaks_4_good_segments = 0,
                    
                    epoch_data_n_clean_autoreject = 0, # epochs data into HEP epochs or 2s epochs to calculate power and complexity markers
                    get_evokeds = 0,
                    grand_average_evokeds = 0,
                    get_average_voltage_HEP = 0,
                    plot_compare_evokeds = 0,
                    plot_compare_topography =1,
                    statistics_space_time = 0,
                    plot_significant_electrodes = 0,
                    
                    get_ecg_parameters = 0,
                    ecg_param_stats_n_plot = 0,
                    
                    get_eeg_markers = 0,
                    markers_stats_n_plot = 1,       
                    
                    final_summary_df = 0, 
                    HEP_markers_corr = 0)

#%%

###############################################################################
#              RUN OPERATIONS OVER SELECTED FILES AND GROUPS
###############################################################################
today = str(date.today())

for file_id in files_id[files_to_run[0]:files_to_run[1]]:
    
    file_index = files_id.index(file_id) 
    print(file_index)

#=========================================================================
# CREATE PATIENT INFO DF
#=========================================================================
    if operations_to_apply['create_patient_info']:
        
        chdir(cfg.outcome_path)
        outcome = glob.glob('*221006.csv')[0]
        chdir(cfg.raw_path)
        file = glob.glob(file_id+'*.edf')[0]  
        create_patient_info(file,file_id,outcome,cfg)    


#=========================================================================
# FILTER - PICK EEG and ECG channels
#=========================================================================..
    if operations_to_apply['filter_raw_hp_lp']:
        
        chdir(cfg.raw_path)
        file = glob.glob(file_id+'*.edf')[0]
        filter_raw_hp_lp(file,file_id,cfg,overwrite)
        
        
#=========================================================================
# Find good ECG - EEG segments - at least 10 minutes
#=========================================================================
    if operations_to_apply['find_R_Peaks_4_good_segments']:
        
        chdir(cfg.filtered_files_path)
        file = glob.glob(file_id+'*notch_raw.fif')[0]  
        subject_info = find_R_Peaks_4_good_segments(file,file_id,file_index,
                                                    plot_R_detection,visual_check,
                                                    cfg)   
 
#=========================================================================
# EPOCH DATA AND CLEAN w AUTOREJECT
#=========================================================================
    if operations_to_apply['epoch_data_n_clean_autoreject']:
        
        chdir(cfg.annot_files_path)
        file = glob.glob(file_id+'*good_raw.fif')[0]

        epochs = epoch_data_n_clean_autoreject(file,file_id,epoch_type,cfg,manual_ch_rejection=manual_ch_rejection,overwrite=overwrite)          
        
        
#=========================================================================
#  GET EVOKEDS PER SUBJECT
#=========================================================================
    if operations_to_apply['get_evokeds'] and epoch_type=='HEP':
        
        epochs_path = cfg.epochs_path_HEP
   
        chdir(epochs_path)
        
        files = glob.glob(file_id+'*-epo.fif')
        if len(files)>0:
            for file in files:
                get_evokeds(file,group_by,cfg,plot=plot,overwrite=overwrite)
        
#=========================================================================
#  GET AVERAGE VOLTAGE FOR HEP 
#=========================================================================
    if operations_to_apply['get_average_voltage_HEP'] and files_id[file_index] == files_id[-1] and epoch_type=='HEP':
        
        chdir(cfg.epochs_path_HEP)
        files = glob.glob(('*-epo.fif'))
        files.sort()
    
        
        for file in files:
            mean_HEP_all_subj = get_average_voltage_HEP(file,files,mean_HEP_all_subj,cfg)
        
        
#=========================================================================
#  GET GRAND AVERAGE OF EVOKEDS
#=========================================================================
    if operations_to_apply['grand_average_evokeds'] and files_id[file_index] == files_id[-1] and epoch_type =='HEP':
        
        conditions = list(cfg.events_dict[group_by]['event_id'].keys())
        evoked_data_all = dict.fromkeys(conditions)
        epochs_included = []
        n_epochs = []
        
        for c in conditions:
            evoked_data_all[c]=[] 

        
        chdir(cfg.averages_path)

        for patient in ids:
            files = [f for f in glob.glob((patient +'_'+'*y-ave.fif'))  if any(substring in f for substring in conditions)]
            if len(files)==0:
                files = [f for f in glob.glob((patient +'_' +'*yn-ave.fif'))  if any(substring in f for substring in conditions)]
            if len(files)!=0:
                files.sort()
                file=files[0] # take the first file           

                epochs_included.append(file)
                evoked_data = mne.read_evokeds(file)
                
                n_epochs.append(evoked_data[0].nave)
                for evoked in evoked_data:
                    trial_type = evoked.comment
                    evoked_data_all[trial_type].append(evoked) 
      
        files_number_epochs=grand_average_evokeds(conditions,group_by,evoked_data_all,cfg,epochs_included, n_epochs,overwrite=overwrite)
        condition1 = files_number_epochs[files_number_epochs['epochs_file'].str.contains(conditions[0])]
        condition1.reset_index(inplace=True)
        condition2 = files_number_epochs[files_number_epochs['epochs_file'].str.contains(conditions[1])]
        condition2.reset_index(inplace=True)
#=========================================================================
# PLOT EVOKED ERP WITH CI FOR SPECIFIC CHANNEL
#=========================================================================
    if operations_to_apply['plot_compare_evokeds'] and files_id[file_index] == files_id[-1] and epoch_type =='HEP':
       
        conditions = list(cfg.events_dict[group_by]['event_id'].keys())
        evoked_data_all = dict.fromkeys(conditions)
        epochs_included = []
        n_epochs = []
        
        for c in conditions:
            evoked_data_all[c]=[] 

        
        chdir(cfg.averages_path)

        for patient in ids:
            files = [f for f in glob.glob((patient +'_'+'*y-ave.fif'))  if any(substring in f for substring in conditions)]
            if len(files)==0:
                files = [f for f in glob.glob((patient +'_' +'*yn-ave.fif'))  if any(substring in f for substring in conditions)]
            if len(files)!=0:
                files.sort()
                file=files[0] # take the first file           

                epochs_included.append(file)
                evoked_data = mne.read_evokeds(file)
                
                n_epochs.append(evoked_data[0].nave)
                for evoked in evoked_data:
                    trial_type = evoked.comment
                    evoked_data_all[trial_type].append(evoked) 
                    
        channel = 'Pz' 
        
        plot_compare_evokeds(evoked_data_all,epochs_included,n_epochs,group_by,conditions,channel,cfg,
                                        trace_plot='yes',overwrite=overwrite)
#=========================================================================
# PLOT TOPOGRAPHY 
#=========================================================================
    if operations_to_apply['plot_compare_topography'] and files_id[file_index] == files_id[-1] and epoch_type =='HEP' :
       
        conditions = list(cfg.events_dict[group_by]['event_id'].keys())
        evoked_data_all = dict.fromkeys(conditions)
        epochs_included = []
        n_epochs = []
        
        for c in conditions:
            evoked_data_all[c]=[] 

        
        chdir(cfg.averages_path)

        for patient in ids:
            files = [f for f in glob.glob((patient +'_'+'*y-ave.fif'))  if any(substring in f for substring in conditions)]
            if len(files)==0:
                files = [f for f in glob.glob((patient +'_' +'*yn-ave.fif'))  if any(substring in f for substring in conditions)]
            if len(files)!=0:
                files.sort()
                file=files[0] # take the first file           

                epochs_included.append(file)
                evoked_data = mne.read_evokeds(file)
                
                n_epochs.append(evoked_data[0].nave)
                for evoked in evoked_data:
                    trial_type = evoked.comment
                    evoked_data_all[trial_type].append(evoked)             
        plot_compare_topography(evoked_data_all,group_by,conditions,cfg,
                                       overwrite=overwrite)
        
        
#=========================================================================
#  CLUSTER PERM ANALYSIS TIME DOMAIN
#=========================================================================
    if operations_to_apply['statistics_space_time'] and files_id[file_index] == files_id[-1] and epoch_type =='HEP':
        
        conditions = list(cfg.events_dict[group_by]['event_id'].keys())
        evoked_data_all = dict.fromkeys(conditions)
        epochs_included = []
        n_epochs = []
        
        for c in conditions:
            evoked_data_all[c]=[] 

        
        chdir(cfg.averages_path)

        for patient in ids:
            files = [f for f in glob.glob((patient +'_'+'*y-ave.fif'))  if any(substring in f for substring in conditions)]
            if len(files)==0:
                files = [f for f in glob.glob((patient +'_' +'*yn-ave.fif'))  if any(substring in f for substring in conditions)]
            if len(files)!=0:
                files.sort()
                file=files[0] # take the first file           

                epochs_included.append(file)
                evoked_data = mne.read_evokeds(file)
                
                n_epochs.append(evoked_data[0].nave)
                for evoked in evoked_data:
                    trial_type = evoked.comment
                    evoked_data_all[trial_type].append(evoked)                  
       
        cluster_dict = statistics_space_time(group_by,today,evoked_data_all,
                                                          conditions,
                                                          cfg,
                                                          overwrite=overwrite)        
#=========================================================================
# PLOT AVERAGE VOLTAGE FOR ELECTRODES IN SIGNIFICANT CLUSTER
#=========================================================================
    if operations_to_apply['plot_significant_electrodes'] and files_id[file_index] == files_id[-1] and epoch_type =='HEP':
        
        conditions = list(cfg.events_dict[group_by]['event_id'].keys())
        evoked_data_all = dict.fromkeys(conditions)
        epochs_included = []
        n_epochs = []
        
        for c in conditions:
            evoked_data_all[c]=[] 

        
        chdir(cfg.averages_path)

        for patient in ids:
            files = [f for f in glob.glob((patient +'_'+'*y-ave.fif'))  if any(substring in f for substring in conditions)]
            if len(files)==0:
                files = [f for f in glob.glob((patient +'_' +'*yn-ave.fif'))  if any(substring in f for substring in conditions)]
            if len(files)!=0:
                files.sort()
                file=files[0] # take the first file           

                epochs_included.append(file)
                evoked_data = mne.read_evokeds(file)
                
                n_epochs.append(evoked_data[0].nave)
                for evoked in evoked_data:
                    trial_type = evoked.comment
                    evoked_data_all[trial_type].append(evoked) 
                    
        chdir(cfg.statistics_path)         
        cluster_info =  glob.glob(('HEP*' +group_by +'*'+'.cluster'))[0]
        print(cluster_info)
 
        plot_significant_electrodes(evoked_data_all,epoch_type,group_by,
                                                          cfg,
                                                          cluster_info)        




#=========================================================================
# GET EEG MARKERS
#=========================================================================
    if operations_to_apply['get_eeg_markers'] and files_id[file_index] == files_id[-1] and epoch_type=='markers':
       
        chdir(cfg.epochs_path_markers)
        files = glob.glob(('*-epo.fif'))
        files.sort()  

        tmin = None # all epoch
        tmax = None # all epoch
        
        for file in files: 

            df_all_subj,df_epochs_all_subj = get_eeg_markers(file,files,tmin,tmax, df_all_subj,df_epochs_all_subj,cfg)  

#=========================================================================
# Markers STATs and PLOTs
#=========================================================================
    if operations_to_apply['markers_stats_n_plot'] and files_id[file_index] == files_id[-1]:
       
           
        chdir(cfg.markers_path)
        all_subj_markers = glob.glob('*markers_all_subj.pkl')[0]
        all_subj_epochs_markers = glob.glob('*markers_epochs_all_subj.pkl')[0]

       

        df_stats = markers_stats_n_plot(group_by,cfg.markers_list,all_subj_markers,all_subj_epochs_markers,cfg)  


#=========================================================================
# ECG PARAMETERS: HR and HRV for each epoch and subject
#=========================================================================
    if operations_to_apply['get_ecg_parameters']:
       
           
        chdir(cfg.annot_files_path)
        file = glob.glob(file_id+'*good_raw.fif')[0]      
        ecg_all_subj = get_ecg_parameters(file,file_id,files_id,file_index,cfg,ecg_all_subj,overwrite=overwrite)

#=========================================================================
# ECG stats and plot
#=========================================================================
    if operations_to_apply['ecg_param_stats_n_plot'] and files_id[file_index] == files_id[-1]:
       
           
        chdir(cfg.ecg_param_path)
        df_file = glob.glob('ecg_all_subj.pkl')[0]      

        ecg_param_stats_n_plot(df_file,group_by,cfg,overwrite=overwrite)

#=========================================================================
# Get DF with summary for all data
#=========================================================================
    if operations_to_apply['final_summary_df'] and files_id[file_index] == files_id[-1]:
       
        chdir(cfg.ecg_param_path)
        ecg_params_file = glob.glob('*category.pkl')[0]      
        chdir(cfg.markers_path)
        markers_file = glob.glob('df_markers_all_subj_w_category.pkl')[0] 
        chdir(cfg.summary_path)
        HEP_amplitude_file = glob.glob('*all_subj.pkl')[0] 
        
        df_final = final_summary_df(HEP_amplitude_file,ecg_params_file,markers_file,cfg)

#=========================================================================
# Correlation between HEP amplitude and power and complexity markers
#=========================================================================
    if operations_to_apply['HEP_markers_corr'] and files_id[file_index] == files_id[-1]:
       
        chdir(cfg.summary_path)
        df_summary = glob.glob('HR_HEP_and_markers.pkl')[0]         
        HEP_markers_corr(df_summary,group_by,cfg)