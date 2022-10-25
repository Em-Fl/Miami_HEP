#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Em-Fl (emilia.flo.rama at gmail)

## BEFORE IMPORTING configs_Miami.py ####

## 1) create 'raw' folder with all .edf files 
## 2) create 'outcomes' folder with .csv with patients outcomes
## 3) create 'scripts' folder with all .py files
## 4) project_path should correspond to path of 'raw', 'outcomes' and 'scripts' folders

"""
import numpy as np
import pandas as pd
from os.path import join, isfile, isdir
from os import makedirs, listdir, environ,walk


###############################################################################
#                                PATHS
##############################################################################

## CHANGE PATH  AND RUN SCRIPT!! 
project_path ='/DATA1/Dropbox/PhD/Project_HEP/Miami_HEP/'


raw_path = join(project_path,'raw')
outcome_path = join(project_path,'outcomes')
functions_path = join(project_path,'scripts')
ecg_param_path = join(project_path,'ecg_params')
images_path = join(project_path,'images')
statistics_path = join(project_path,'statistics')
markers_path = join(project_path,'eeg_markers')
summary_path = join(project_path,'data_summary')


preprocessing_path = join(project_path,'preprocessing')
filtered_files_path = join(preprocessing_path,'filtered_files')
annot_files_path = join(preprocessing_path,'annot_good')
info_path = join(preprocessing_path,'info')
epochs_path_HEP = join(preprocessing_path,'epochs_HEP')
epochs_path_markers = join(preprocessing_path,'epochs_markers')
averages_path = join(preprocessing_path,'averages')
GA_path = join(preprocessing_path,'grand_averages')


#=========================================================================
#  CREATES ALL FOLDERS FOR PROJECT (except: raw,outcomes,scripts)
#=========================================================================
   
def populate_data_directory(project_path,preprocessing_path):
       
            
    project_subfolders=['data_summary','ecg_params','eeg_markers',
                        'images','statistics1','preprocessing']
    
    for folder in project_subfolders:
        full_path_folder = join(project_path,folder)
            ## create figure paths
        try:
            makedirs(full_path_folder)
            print(full_path_folder + ' has been created')
        except OSError as exc:
            if exc.errno == 17: ## dir already exists
                pass
                
    preprocessing_subfolders =['epochs_markers','epochs_HEP','averages',
                               'grand_averages','info','filtered_files',
                               'annot_good']
    
    for folder in preprocessing_subfolders:
        full_path_folder = join(preprocessing_path,folder)
            ## create figure paths
        try:
            makedirs(full_path_folder)
            print(full_path_folder + ' has been created')
        except OSError as exc:
            if exc.errno == 17: ## dir already exists
                pass
    
if not isdir(preprocessing_path):    
    populate_data_directory(project_path,preprocessing_path)   
 
###############################################################################
###############################################################################
###############################################################################
#                           PREPROCESSING PARAMETERS
###############################################################################
###############################################################################
###############################################################################


###############################################################################
#                                MONTAGES
###############################################################################
Montage1 ={}
Montage1['all_ch'] = ['Event','C3', 'C4', 'O1', 'O2', 'A1', 'A2', 'Cz', 'F3', 'F4', 'F7',
            'F8', 'Fz', 'Fp1', 'Fp2', 'Fpz', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 
            'T6', 'LOC', 'ROC', 'CHIN1', 'CHIN2', 'ECGL', 'ECGR', 'LAT1', 'LAT2',
            'RAT1', 'RAT2', 'CHEST', 'ABD', 'FLOW', 'SNORE', 'DIF5', 'DIF6', 
            'POS', 'DC2', 'DC3', 'DC4', 'DC5', 'DC6', 'DC7', 'DC8', 'DC9',
            'DC10', 'OSAT', 'PR']
Montage1['eeg_ch'] = ['C3','C4','O1','O2','Cz','F3','F4','F7','F8','Fz','Fp1','Fp2','Fpz',
                        'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']
Montage1['ecg_ch'] =['ECGL','ECGR']
 
Montage2 = {}                                             
Montage2['all_ch'] = ['Event','C3', 'C4', 'O1', 'O2', 'A1', 'A2', 'Cz', 'F3', 'F4', 'F7',
            'F8', 'Fz', 'Fp1', 'Fp2', 'Fpz', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 
            'T6', 'LOC', 'ROC', 'CHIN1', 'CHIN2', 'ECGL', 'ECGR', 'LAT1', 'LAT2',
            'RAT1', 'RAT2', 'CHEST', 'ABD', 'FLOW', 'SNORE', 'DIF5', 'DIF6', 
            'POS', 'DC2', 'DC3', 'DC4', 'DC5', 'DC6', 'DC7', 'DC8', 'DC9',
            'DC10', 'OSAT']
Montage2['eeg_ch'] = ['C3','C4','O1','O2','Cz','F3','F4','F7','F8','Fz','Fp1','Fp2','Fpz',
                        'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']
Montage2['ecg_ch'] =['ECGL','ECGR']

###############################################################################
#                                   FILTER PARAMS
###############################################################################
notch = 60
hp = 0.5
lp = 40
method ='fir'
phase = 'zero'

###############################################################################
#                              EPOCH PARAM
###############################################################################

minimum_epochs_HEP = 400 # minimum epochs after cleaning in order to consider segment for evokeds
# minimum_epochs_markers = 100 # minimum epochs after cleaning in order to consider segment for markers


epoch_param = {'HEP':{},'markers':{}}
epoch_param['HEP']['baseline'] = None
epoch_param['HEP']['tmin'] = -0.2
epoch_param['HEP']['tmax'] = 0.8
epoch_param['HEP']['detrend'] = 1

epoch_param['HEP']['n_interpolates'] = np.array([1, 4, 6, 10, 15])
epoch_param['HEP']['consensus_percs'] =  np.linspace(0, 1.0, 11)
epoch_param['HEP']['n_bads_threshold'] = 0.70

epoch_param['markers']['baseline'] = None
epoch_param['markers']['tmin'] = 0
epoch_param['markers']['tmax'] = 2
epoch_param['markers']['detrend'] = None

epoch_param['markers']['n_epo_segments'] = int(60*10/(int(epoch_param['markers']['tmax']-epoch_param['markers']['tmin'])))
epoch_param['markers']['n_interpolates'] = np.array([1, 4, 6, 10, 15])
epoch_param['markers']['consensus_percs'] =  np.linspace(0, 1.0, 11)
epoch_param['markers']['n_bads_threshold'] = 0.70

   
###############################################################################
#                              EVENTS INFO
###############################################################################
events_dict = {'outcome':{},'gose_dc':{},'command_score_dc':{}}

events_dict['outcome']['event_id'] ={ 'alive':1,'dead':0}
events_dict['command_score_dc']['event_id'] = {'good_cs_dc':1,'bad_cs_dc':0}
events_dict['command_score_dc']['good'] = [4,5,6]
events_dict['gose_dc']['event_id'] = {'good_gose':1,'bad_gose':1}
events_dict['gose_dc']['good']= [4,5,6,7,8]

###############################################################################
#                              CLUSTER PERM PARAMS
###############################################################################

cl_stat_dict = {}

cl_stat_dict['time_window'] = [0,0.6+0.005]
cl_stat_dict['n_permutations'] = 2000
cl_stat_dict['max_step'] = 1
cl_stat_dict['init_p_value'] = 0.05
cl_stat_dict['tail'] = 0
cluster_p_value = 0.05

###############################################################################
#                             DF INIT
###############################################################################

# ECG Init DF 
df_ecg = {'file_id':[],
     'segment_number':[],
     'tmin_s':[], 
     'tmax_s':[],
     'ecg_segment_good':[],
     'eeg_segment_good':[],
     'outcome':[], 
     'gose_dc':[],
     'command_score_dc':[],
     "ECG_Rate_Mean": [],
     "HRV_RMSSD": [],
     "HRV_MeanNN": [],
     "HRV_SDSD": []}


column_names_markers = ['file_id','segment_number','tmin_s','tmax_s','n_epochs_for_markers','eeg_segment_good','ecg_segment_good','outcome','gose_dc','command_score_dc','kolcom', 'p_e', 'wSMI', 
                'delta', 'delta_n', 'theta', 'theta_n', 
                'alpha', 'alpha_n', 'beta', 'beta_n' ]


#HEP average voltage df
columns_HEP = ['file_id','segment_number', 'number_epochs', 'HEP_data_good',
 'C3_0_600',
 'C4_0_600',
 'O1_0_600',
 'O2_0_600',
 'Cz_0_600',
 'F3_0_600',
 'F4_0_600',
 'F7_0_600',
 'F8_0_600',
 'Fz_0_600',
 'Fp1_0_600',
 'Fp2_0_600',
 'Fpz_0_600',
 'P3_0_600',
 'P4_0_600',
 'Pz_0_600',
 
 'C3_600_800',
 'C4_600_800',
 'O1_600_800',
 'O2_600_800',
 'Cz_600_800',
 'F3_600_800',
 'F4_600_800',
 'F7_600_800',
 'F8_600_800',
 'Fz_600_800',
 'Fp1_600_800',
 'Fp2_600_800',
 'Fpz_600_800',
 'P3_600_800',
 'P4_600_800',
 'Pz_600_800',
 
 'C3_200_450',
  'C4_200_450',
  'O1_200_450',
  'O2_200_450',
  'Cz_200_450',
  'F3_200_450',
  'F4_200_450',
  'F7_200_450',
  'F8_200_450',
  'Fz_200_450',
  'Fp1_200_450',
  'Fp2_200_450',
  'Fpz_200_450',
  'P3_200_450',
  'P4_200_450',
  'Pz_200_450']

##############################################################################
frontal_ch = ['Fp1','Fpz','Fp2','F7',
              'F3','Fz','F4','F8']
central_post_ch = ['C3','Cz','C4','P3',
              'Fz','P4','O1','O2']
markers_list = ['kolcom', 'p_e', 'wSMI', 'delta_n', 'theta_n',
                 'alpha_n',  'beta_n']

marker_names = {'kolcom':'Kolmog complexity', 
                'p_e':'permutation entropy',
                'wSMI':'wSMI',
                'delta_n':'delta norm',
                'theta_n':'theta norm',
                'alpha_n':'alpha norm',
                'beta_n':'beta norm'}

grouping_names = {'command_score_dc_category':'command score', 
                'gose_dc_category':'gose',
                'outcome':'outcome',
                'command_score_dc':'command score', 
                                'gose_dc':'gose'}
###############################################################################
#                            PLOT PARAMETERS
###############################################################################

palette = ['darkcyan','darkorchid']
linewidth=3

