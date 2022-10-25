#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Em-Fl (emilia.flo.rama at gmail)

"""
import pandas as pd
from os.path import join

#========================================================================
#
# DATA 
# HEP average voltage
# markers 
# Heart activity variables

#=========================================================================        
def final_summary_df(HEP_amplitude_file,ecg_params_file,markers_file,cfg):
    
    HEP_amplitude = pd.read_pickle(join(cfg.summary_path,HEP_amplitude_file))
    HEP_amplitude.rename(columns={'id_segment_number':'id_segment'},inplace=True)
    
    ecg_params = pd.read_pickle(join(cfg.ecg_param_path,ecg_params_file))
    ecg_params['segment_number']= ecg_params['segment_number'].astype(int).astype(str)
    ecg_params.drop(labels=['id'],axis=1,inplace=True)    
    
    markers = pd.read_pickle(join(cfg.markers_path,markers_file))
    
    
    ecg_params['file_id_segment'] = ecg_params['file_id']+'_'+ecg_params['segment_number']
    HEP_amplitude['file_id_segment'] = HEP_amplitude['file_id']+'_'+HEP_amplitude['segment_number']
    markers['file_id_segment'] = markers['file_id']+'_'+markers['segment_number']

    
    diff = list(set(HEP_amplitude['file_id_segment']) - set(ecg_params['file_id_segment'] ))
    
    if len(diff)>0:
        print('number of segments used for HEP and to calculate HR do not match')


    columns = ecg_params.columns.to_list()
    columns_new = []
    columns_new.append(columns[-1])
    columns_new.extend( columns[:-1])
    ecg_params = ecg_params[columns_new]
    
    columns = HEP_amplitude.columns.to_list()
    columns_new = []
    columns_new.append(columns[-1])
    columns_new.extend( columns[:-1])
    HEP_amplitude = HEP_amplitude[columns_new]
    
    columns = markers.columns.to_list()
    columns_new = []
    columns_new.append(columns[-1])
    columns_new.extend( columns[:-1])
    markers = markers[columns_new]


   
    ecg_params['outcome'] = ecg_params['outcome'].astype(int)
    ecg_params['gose_dc'] = ecg_params['gose_dc'].astype(int)
    ecg_params['command_score_dc'] = ecg_params['command_score_dc'].astype(int)
  
    
    markers['outcome'] = markers['outcome'].astype(int)
    markers['gose_dc'] = markers['gose_dc'].astype(int)
    markers['command_score_dc'] = markers['command_score_dc'].astype(int)
    
    
    HEP_amplitude.drop(labels=['file_id','segment_number'],axis=1,inplace=True)
    merged_df = ecg_params.merge(HEP_amplitude, how = 'outer', on = ['file_id_segment'])
    merged_df2 = merged_df.merge(markers, how = 'outer', 
                                 right_on = ['file_id_segment','outcome',
                                             'gose_dc','command_score_dc',
                                             'file_id','segment_number',
                                             'eeg_segment_good','ecg_segment_good',
                                             'outcome_category', 
                                             'gose_dc_category', 'command_score_dc_category','tmin_s','tmax_s'],
                                 left_on =  ['file_id_segment','outcome',
                                             'gose_dc','command_score_dc',
                                             'file_id','segment_number',
                                             'eeg_segment_good','ecg_segment_good',
                                             'outcome_category', 
                                             'gose_dc_category', 'command_score_dc_category','tmin_s','tmax_s'])
    

            
    merged_df2.to_pickle(join(cfg.summary_path,'HR_HEP_and_markers.pkl'),protocol=4) 
    merged_df2.to_csv(join(cfg.summary_path,'HR_HEP_and_markers.csv'))
    
    return merged_df2
    
    
    


    

    
    
