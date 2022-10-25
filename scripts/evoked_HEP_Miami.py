#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:15:16 2022

@author: emilia.ramaflo
"""
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import mne
import pickle
from datetime import date

#========================================================================
#
# EVOKED DATA
# Average all epochs in 10' segment if after cleaning there are more than 
# cfg.minimum_epochs_HEP, and the data was marked as y or yn after 'epoch_data_n_clean_autoreject'
#=========================================================================      

def get_evokeds(file,group_by,cfg,plot,overwrite):
    
    
    plt.close('all')
    
    subject_info_dict_path = join(cfg.info_path,file[:12]+'_info.pkl')
    subject_info_open = open(subject_info_dict_path , "rb")
    subject_info = pickle.load(subject_info_open) 
    chunk_ok_data = int(file.split('_')[2][:-8])

    if subject_info['R_info'][chunk_ok_data]['HEP_data_good']=='y' or subject_info['R_info'][chunk_ok_data]['HEP_data_good']=='yn': 
        if overwrite:
            
            print('\n evoked for file ' + file + '\n')
            
            epochs = mne.read_epochs(file)
            
            if len(epochs)>cfg.minimum_epochs_HEP:
            
                elecs_extra = ['T3','T4','T5','T6']
                for e in elecs_extra:
                    if e in epochs.info.ch_names:
                        epochs = epochs.drop_channels(e)
                        print('\n dropping channel ',e,'\n')
                    
                event_id = cfg.events_dict[group_by]['event_id'] 
        
                if group_by =='outcome':
                    
                    epochs.events[:,2] = [subject_info['outcome']] * len(epochs)
                    new_key = list(event_id.keys())[list(event_id.values()).index(subject_info['outcome'])]
                    epochs.event_id = {new_key:subject_info['outcome']}    # modify event_id accoording to outcome
                    
                if group_by =='gose_dc':
                    new_key = subject_info['gose_dc_cat']
                    epochs.events[:,2] = [event_id[new_key]] * len(epochs)
                    epochs.event_id = {new_key:event_id[new_key]}
               
                if group_by =='command_score_dc':
                    new_key = subject_info['command_score_dc_cat']
            
                    epochs.events[:,2] = [event_id[new_key]] * len(epochs)
                    epochs.event_id = {new_key:event_id[new_key]}
        
                          
          
                evokeds = epochs.average()
                evokeds_name = file[:-8] + '_' + new_key +'_'+subject_info['R_info'][chunk_ok_data]['HEP_data_good']+'-ave.fif'    
                mne.evoked.write_evokeds(join(cfg.averages_path,evokeds_name), evokeds)
                
                if plot == True:
            
                    evoked_fig = file[:-8] +'_' + new_key +'_'+subject_info['R_info'][chunk_ok_data]['HEP_data_good']+'_evoked.pdf'
                    evoked_fig_path = join(cfg.images_path,evoked_fig)
        
                    fig1, ax1 = plt.subplots(1,1)
        
                    evokeds.plot(axes=ax1,spatial_colors=True,gfp=True)
                    ax1.set_title(file[:-8] +' '+ new_key)
        
                    plt.show()
                    fig1.savefig(evoked_fig_path)
                    plt.pause(1)
                    plt.close()
            else: 
                print('\n less than cfg.minimum_epochs_HEP HEP epochs for this patient segemnt \n')
            
        else:
            print('evokeds file already exists')
    else:
        print('HEP evoked was marked as not good')
        
        
#========================================================================
#
# GET AVERGE VOLTAGE FOR HEP TIME WINDOWS
#
#=========================================================================      

def get_average_voltage_HEP(file,files,mean_HEP_all_subj,cfg):
    
        subject_info_dict_path = join(cfg.info_path,file[:12]+'_info.pkl')
        subject_info_open = open(subject_info_dict_path , "rb")
        subject_info = pickle.load(subject_info_open)  
    
        chunk_ok_data = int(file.split('_')[2][:-8])

        epochs = mne.read_epochs(join(cfg.epochs_path_HEP,file))
        sf = epochs.info['sfreq']
        elecs_extra = ['T3','T4','T5','T6']
        for e in elecs_extra:
            if e in epochs.info.ch_names:
                epochs = epochs.drop_channels(e)
                print('\n dropping channel ',e,'\n')
            
        baseline_samples = int(0.2*sf)
        
        t_200 = int(0.2*sf) + baseline_samples
        t_450 = int(0.45*sf) + baseline_samples
        t_600 = int(0.6*sf) + baseline_samples
        t_800 = int(0.8*sf) + baseline_samples 
        
        n_epochs = len(epochs)
        evoked = epochs.average()
        
        evoked_0_600 = evoked.to_data_frame().loc[baseline_samples:t_600].drop('time',axis=1).mean(axis=0).values
        evoked_600_800 = evoked.to_data_frame().loc[t_600:t_800].drop('time',axis=1).mean(axis=0).values
        evoked_200_450 = evoked.to_data_frame().loc[t_200:t_450].drop('time',axis=1).mean(axis=0).values
        
        t_wave = [e + '_200_450' for e in epochs.info.ch_names]
        early = [e +'_0_600' for e in epochs.info.ch_names]
        late = [e +'_600_800' for e in epochs.info.ch_names]
        
        column_names = ['file_id']
        column_names.extend(['segment_number'])
        column_names.extend(['number_epochs'])
        column_names.extend(['HEP_data_good'])
        column_names.extend(early)
        column_names.extend(late)
        column_names.extend(t_wave)
                              
        data_columns = [file[0:12]]
        data_columns.extend([file[13:-8]])
        data_columns.extend([n_epochs])
        data_columns.extend([subject_info['R_info'][chunk_ok_data]['HEP_data_good']])

        data_columns.extend(evoked_0_600)
        data_columns.extend(evoked_600_800)
        data_columns.extend(evoked_200_450)
        
        df = pd.DataFrame([data_columns], 
                    columns = column_names)

        mean_HEP_all_subj = pd.concat([mean_HEP_all_subj,df])  
        
                                     
        if file == files[-1]:  
            mean_HEP_all_subj = mean_HEP_all_subj.reset_index()
            mean_HEP_all_subj.drop(labels=['index'],axis=1,inplace=True)
            mean_HEP_all_subj.to_pickle(join(cfg.summary_path ,'HEP_average_voltage_all_subj.pkl'),protocol=4)  
        
        return mean_HEP_all_subj
        

                     

#========================================================================
#
# GRAND AVERAGE FROM EVOKED DATA
# plots histogram for number of epochs for each groupping category
#
#=========================================================================         
def grand_average_evokeds(conditions,group_by,evoked_data_all,cfg,epochs_included,n_epochs,overwrite):
    
    conditions = list(evoked_data_all.keys())
    today = str(date.today())
  
    grand_averages = dict()
    
    trial_types=[]
    
    for trial_type in evoked_data_all:
        
        grand_averages[trial_type] = mne.grand_average(evoked_data_all[trial_type])
        grand_averages[trial_type].comment = trial_type  
        
    for trial_type in grand_averages:
        
        trial_types.append(trial_type)
        
            
        grand_average_path = join(cfg.GA_path,trial_type+'_grand_average' + '_'+today+'_'+'-ave.fif')

        mne.evoked.write_evokeds(grand_average_path,
                                 grand_averages[trial_type])
        
    
    df = pd.DataFrame({'epochs_file':epochs_included,'number_epochs':n_epochs})
    df.to_pickle(join(cfg.GA_path,conditions[0]+'_'+conditions[1]+'_files_selected_'+today+'.pkl'))
    condition1 = df[df['epochs_file'].str.contains(conditions[0])]['number_epochs']

    condition2 = df[df['epochs_file'].str.contains(conditions[1])]['number_epochs']
    fig,ax =plt.subplots(1,1)
    plt.hist(condition1, color=cfg.palette[0])
    plt.hist(condition2, color=cfg.palette[1])
    plt.suptitle('number epochs')
    plt.show()
    plt.pause(5)
    fig.savefig(join(cfg.images_path,'histogram_number_epochs_'+group_by +'.pdf'))
    plt.close()
        

    
    return df



