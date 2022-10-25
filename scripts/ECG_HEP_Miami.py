#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Em-Fl (emilia.flo.rama at gmail)

"""
import mne
from os.path import join, isfile
import pandas as pd
import pickle
import neurokit2 as nk



#========================================================================
#
# EVALUATE CARDIAC ACTIVITY FOR EACH EPOCH 
#
#=========================================================================      

def get_ecg_parameters(file,file_id,files_id,file_index,cfg,ecg_all_subj,overwrite):  
    

    file_path_name = join(cfg.annot_files_path,file)

    subject_info_dict_path = join(cfg.info_path,file_id+'_info.pkl')
    
    if isfile(subject_info_dict_path):

        subject_info_open = open(subject_info_dict_path , "rb")
        subject_info = pickle.load(subject_info_open)  
       
                
        if 'R_info' in list(subject_info.keys()):
            
            filtered_file = mne.io.read_raw_fif(file_path_name, preload = True,verbose='ERROR')
            filtered_file.pick_channels(subject_info['ecg_ch'])
                 

            
            for chunk_ok_data in list(subject_info['R_info'].keys()):
                
                if (subject_info['eeg_data_good']=='y' or subject_info['eeg_data_good']=='yn') and (subject_info['R_info'][chunk_ok_data]['ecg_data_good'] =='y' or subject_info['R_info'][chunk_ok_data]['ecg_data_good'] =='yn'):
                    print('finding HR and HRV for:' + str(file_id) + ' segment number '+ str(chunk_ok_data))
                    
                    tmin = subject_info['R_info'][chunk_ok_data]['tmin']
                    tmax = subject_info['R_info'][chunk_ok_data]['tmax']
                    good_data = filtered_file.copy().crop(tmin=tmin,tmax=tmax)
                    ECG = good_data.get_data(start=0, stop=None, return_times=False) # get ECG channel data
                        
                    if subject_info['ecg_subtraction'] == ['ECGL', 'ECGR']:
                        ECG_diff = ECG[1]-ECG[0] 
                    if subject_info['ecg_subtraction'] == ['ECGR', 'ECGL']:  
                        ECG_diff = ECG[0]-ECG[1]
                    if subject_info['ecg_subtraction'] == ['ECGL']: 
                        ECG_diff = ECG[0]
                    if subject_info['ecg_subtraction'] == ['ECGR']: 
                        ECG_diff = ECG[1] 
                    if subject_info['ecg_subtraction'] == ['ECGL_inverted']: 
                        ECG_diff = ECG[0]*-1
                    if subject_info['ecg_subtraction'] == ['ECGR_inverted']: 
                        ECG_diff = ECG[1]*-1               
         
         
                    ecg_signals, info = nk.ecg_process(ECG_diff, sampling_rate=filtered_file.info['sfreq'])
                    ecg_hrv  = nk.ecg_intervalrelated(ecg_signals,sampling_rate=filtered_file.info['sfreq'])
                    
                    # fill d
                    d = {'file_id':file_id,
                         'segment_number':int(chunk_ok_data),
                         'tmin_s':tmin, 
                         'tmax_s':tmax,
                         'ecg_segment_good':subject_info['R_info'][chunk_ok_data]['ecg_data_good'],
                         'eeg_segment_good':subject_info['R_info'][chunk_ok_data]['eeg_data_good'],
                         'outcome':subject_info['outcome'], 
                         'gose_dc':subject_info['gose_dc'],
                         'command_score_dc':subject_info['command_last'],
                         "ECG_Rate_Mean": ecg_hrv['ECG_Rate_Mean'].values[0],
                         "HRV_RMSSD": ecg_hrv['HRV_RMSSD'] .values[0],
                         "HRV_MeanNN": ecg_hrv['HRV_MeanNN'].values[0],
                         "HRV_SDSD": ecg_hrv['HRV_SDSD'].values[0]}
                    
                    ecg_epoch = pd.DataFrame(data=d, index=[0])
    
                    ecg_all_subj = pd.concat([ecg_all_subj,ecg_epoch])
                    
        else:
            d = {'file_id':file_id,
             'segment_number':int(chunk_ok_data),
             'tmin_s':tmin, 
             'tmax_s':tmax,
             'ecg_data_good':'n',
             'eeg_data_good':subject_info['eeg_data_good'],
             'outcome':subject_info['outcome'], 
             'gose_dc':subject_info['gose_dc'],
             'command_score_dc':subject_info['command_last'], 
             "HRV_RMSSD": 'NA',
             "HRV_MeanNN": "NA",
             "HRV_SDSD": 'NA'}    
            
            ecg_epoch = pd.DataFrame(data=d, index=[0])

            ecg_all_subj = pd.concat([ecg_all_subj,ecg_epoch])
    
                                    
        if files_id[file_index] == files_id[-1]:  
            ecg_all_subj=ecg_all_subj.reset_index()
            ecg_all_subj.drop(labels=['index'],axis=1,inplace=True)
            ecg_all_subj.to_pickle(join(cfg.ecg_param_path ,'ecg_all_subj.pkl'),protocol=4)  
            
    return ecg_all_subj
    
        
    

