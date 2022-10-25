#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: emilia.ramaflo
"""
import mne
import pickle
import pandas as pd
import numpy as np
from os import chdir
from os.path import join, isfile
import neurokit2 as nk
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import sys
import copy
import mplcursors
from autoreject import AutoReject


#=========================================================================
# Creates a dictionary for each file with:
# - clinical information (scales and outcome)
# - electrode montage
#=========================================================================
def create_patient_info(file,file_id,outcome,cfg):
    
    info_name = file_id + '_info.pkl'
    # info_name2 = file_id[0:3] + '_info.pkl'
    subject_info_path = join(cfg.info_path,info_name)

    # subject_info_path1 = join(cfg.info_path,info_name)
    # subject_info_path2 = join(cfg.info_path,info_name2)

    if not isfile(join(cfg.info_path,subject_info_path)):
        subject_info ={}
        print('\x1b[1;35m'+'creating subject info df for ' + file_id +'\x1b[1;35m')
        print("\x1b[0m")
    else:
        subject_info_open = open(subject_info_path, "rb")
        subject_info = pickle.load(subject_info_open) 

    outcome_df = pd.read_csv(join(cfg.outcome_path,outcome),sep=',')
    chdir(cfg.raw_path)
    record_id = file_id.split('_')[0]
    
    if record_id == '20~':
        record_id = record_id[:-1]
        
    if record_id == '759 ':
        record_id = record_id[:-1]
    if record_id == 'EBC014 ':
        record_id = record_id[:-1]
    

    row_idx_raw = outcome_df.index[outcome_df['record_id'] == record_id].to_list()
    subject_info['patient_id'] = record_id
    subject_info['file_name'] = file
    subject_info['outcome'] = outcome_df['vital_discharge_0_died_1_alive'][row_idx_raw].to_list()[0]
    subject_info['gose_dc'] = outcome_df['gose_dc'][row_idx_raw].to_list()[0]
    subject_info['command_last'] = outcome_df['command_last'][row_idx_raw].to_list()[0]
    
    if subject_info['gose_dc'] >3:
        subject_info['gose_dc_cat'] = 'good_gose' 
    else:
        subject_info['gose_dc_cat'] = 'bad_gose' 
        
    if subject_info['command_last'] >3:
        subject_info['command_score_dc_cat'] = 'good_cs_dc' 
    else:
        subject_info['command_score_dc_cat'] = 'bad_cs_dc' 

    if 'eeg_ch' not in subject_info:
        raw =  mne.io.read_raw_edf(file, preload = True,verbose='ERROR')
    
        if raw.info.ch_names == cfg.Montage1['all_ch']:
            subject_info['eeg_ch'] = cfg.Montage1['eeg_ch']
            subject_info['ecg_ch'] = cfg.Montage1['ecg_ch']
        else:
            subject_info['eeg_ch'] = cfg.Montage2['eeg_ch']
            subject_info['ecg_ch'] = cfg.Montage2['ecg_ch']
    
        
        raw_= raw.copy().pick(picks = subject_info['eeg_ch'] + subject_info['ecg_ch'])
        
        
        annot_flat = mne.preprocessing.annotate_flat(raw_, bad_percent=99,
                                                      min_duration=0.1, verbose=True)

        
        subject_info['annotations'] = annot_flat  
     
    subject_info_path = join(cfg.info_path,file_id+'_info.pkl')  # NOTE: this will duplicate patient info files with EBC...delete old ones.  
    pickle.dump(subject_info,open(subject_info_path, "wb"))
        

#=========================================================================
# Apply LP and HP to raw continous data
# Apply 60Hz Notch filter for AC
# define channel subtraction to detect R peaks

# l_freq and h_freq are the frequencies below which and above which,
# respectively, to filter out of the data. Thus the uses are:

# - l_freq < h_freq: band-pass filter

# - l_freq > h_freq: band-stop filter

# - l_freq is not None and h_freq is None: high-pass filter

# - l_freq is None and h_freq is not None: low-pass filter
#=========================================================================
def filter_raw_hp_lp(file,file_id,cfg,overwrite):
    
    file_path = join(cfg.raw_path,file)        
        
    if overwrite:
        
        print('\x1b[1;35m'+'filtering raw file ' + file +'\x1b[1;35m')
        print("\x1b[0m")   
        
        raw =  mne.io.read_raw_edf(file_path, preload = True,verbose='ERROR')
        subject_info_dict_path = join(cfg.info_path,file_id+'_info.pkl')
        subject_info_open = open(subject_info_dict_path , "rb")
        subject_info = pickle.load(subject_info_open)  
 
        raw.pick(picks = subject_info['eeg_ch']+ subject_info['ecg_ch'])
        raw.notch_filter(60)
        raw.set_channel_types({raw.info.ch_names[-1]:'ecg'})   
        raw.set_channel_types({raw.info.ch_names[-2]:'ecg'})  
        if type(subject_info['annotations']) is tuple:
            raw.set_annotations(subject_info['annotations'][0])
        else:
            raw.set_annotations(subject_info['annotations'])

            
        if cfg.hp != None and cfg.lp != None:     
            raw.filter(l_freq=cfg.hp,h_freq=cfg.lp,method=cfg.method,phase=cfg.phase) 
        else:
            raw.filter(l_freq=cfg.lp,h_freq=cfg.hp,method=cfg.method,phase=cfg.phase)
            
        filter_name = file_id+ '_' + str(raw.info['highpass']) + '_' + str(raw.info['lowpass'])+'_60Hz_notch' + '_raw.fif'
        filter_path = join(cfg.filtered_files_path, filter_name)   
                      
        raw.save(filter_path, overwrite=True)

            
        if 'ecg_subtraction' not in subject_info:
            
            print('Prepare to select order of bipolar subtraction')
            raw.set_channel_types({raw.info.ch_names[-1]:'eeg'})   
            raw.set_channel_types({raw.info.ch_names[-2]:'eeg'}) 
            rawLR = raw.copy().set_eeg_reference(ref_channels=[raw.info.ch_names[-2]]).pick(raw.info.ch_names[-1])
            rawRL = raw.copy().set_eeg_reference(ref_channels=[raw.info.ch_names[-1]]).pick(raw.info.ch_names[-2])
            rawRL.rename_channels({rawRL.info.ch_names[0]:'R-L'})
            rawLR.rename_channels({rawLR.info.ch_names[0]:'L-R'})
            raw_ecg = raw.copy().pick(raw.info.ch_names[-2:])
            rawRL.add_channels([rawLR])
            rawRL.plot(block=True,duration=4,scalings =dict(eeg=300e-6))
    
            choice = input('ECG channel subtraction LR, RL, n?: ') 
            if choice == 'LR':
                subject_info['ecg_subtraction'] = [raw.info.ch_names[-2], raw.info.ch_names[-1]]
            elif choice=='RL':
                subject_info['ecg_subtraction'] = [raw.info.ch_names[-1], raw.info.ch_names[-2]]
            else:
                raw_ecg.plot(block=True,duration=4,scalings =dict(eeg=300e-6))   
                choice = input('ECG channel good (LR ,RL, L,R, L-,R- for inverted, n neither): ') 
                if choice == 'L':
                    subject_info['ecg_subtraction'] = [raw.info.ch_names[-2]]
                if choice =='R':
                    subject_info['ecg_subtraction'] = [raw.info.ch_names[-1]]
                if choice == 'L-':
                    subject_info['ecg_subtraction'] = [raw.info.ch_names[-2]+'_inverted']
                if choice =='R-':
                    subject_info['ecg_subtraction'] = [raw.info.ch_names[-1]+'_inverted']
                if choice =='LR':
                    subject_info['ecg_subtraction'] = [raw.info.ch_names[-2], raw.info.ch_names[-1]]
                if choice=='RL':
                    subject_info['ecg_subtraction'] = [raw.info.ch_names[-1], raw.info.ch_names[-2]] 
                if choice=='n':
                    subject_info['ecg_subtraction'] = 'no ECG signal' 
            
                    
        
                
        pickle.dump(subject_info,open(subject_info_dict_path, "wb"))

            
            
            


#=========================================================================
# - Detects at least 10 minutes of good data (data that wasn't annotated as bad)
# - Finds R peaks for good segments
# - Saves R_Peaks information in patient info dictionary
# - Saves R Peaks events (array len(R_Peaks)*3), such that sample of R peaks corresponds to 
#    to sample of raw file and not related to the good segment onset 
#    (can be used directly to epoch raw data)
#=========================================================================        
def find_R_Peaks_4_good_segments(file,file_id,file_index,plot_R_detection,visual_check,cfg):
    
    onsets =[] # to fill with info on good segments and add to annotations in raw
    offsets =[]
    durations = []
    description = []
    
    good_segments = [] 
    
    subject_info_dict_path = join(cfg.info_path,file_id+'_info.pkl')
    subject_info_open = open(subject_info_dict_path , "rb")
    subject_info = pickle.load(subject_info_open)  
    
                
    if subject_info['ecg_subtraction'] =='no ECG signal':
        print('no ECG signal')
        print(file_id)
        print('file_index: ' +str(file_index))
        sys.exit()

    filt_file = mne.io.read_raw_fif(join(cfg.filtered_files_path,file), preload = True) # read filtered file

    
    if 'eeg_data_good' not in subject_info.keys(): # already analysed this file and the filtered file already has good and bad segments marked
        print('\x1b[1;35m'+'finding 10 min of good eeg-ecg data for ' + file_id +'\x1b[1;35m')
        print("\x1b[0m")   
    
    
        onset_bad_segments = filt_file.annotations.onset.tolist() # get onset of bad segments in seconds
        duration_bad_segments = filt_file.annotations.duration.tolist() # get duration of bad segments in seconds
        offset_bad_segments = [on+duration for (on, duration) in zip(onset_bad_segments,duration_bad_segments)] #calculate offset of bad segments
        all_ = np.sort(onset_bad_segments +offset_bad_segments) # join onsets and offsets
        all_diff = np.diff(all_)/60  # get differences of time between onsets and offsets in minutes
        index_len_bad_good_segments = np.where((all_diff>10))[0].tolist() # find periods of more than 10 minutes
        
        tiempos=[]    
    
        if onset_bad_segments ==[]: # if no bad segments divide time of recording in 10 min segments, check if offset of segment is not outisde time of recording when considering 10s more to avoid edge artifacts
            print('no bad segments')
            for idx in range(0, round(round(filt_file.times[-1])/600)): 
                onset = filt_file.times[0] + 10 +  idx*60*10
                offset = filt_file.times[0] + 10*60+10 + idx*60*10# 10 seconds to avoid edge artifacts
                if offset < filt_file.times[-1]:
                    tiempos.append([onset, offset])
                    good_segments.append(idx)
                else:
                    continue
        elif (onset_bad_segments[0]-10)/600>=1: #find good segments before onset of first bad segment
            for idx in range(0, round(round(onset_bad_segments[0]-10)/600)): 
                onset = filt_file.times[0] + 10 +  idx*60*10
                offset = filt_file.times[0] + 10*60+10 + idx*60*10# 10 seconds to avoid edge artifacts
                
              
                if offset < onset_bad_segments[0]-10:
                    tiempos.append([onset, offset])
                    good_segments.append(idx)
                    
                    
                    
                else:
                    continue    
        
        elif round(filt_file.times[-1]) ==round(offset_bad_segments[0]) and len(onset_bad_segments)==1:
            print('one bad segment at the end of file (eeg disconected)')
                      
        
    
            for idx in range(0, round(round(onset_bad_segments[0])-10/600)): 
                onset = filt_file.times[0] + 10 +  idx*60*10
                offset = filt_file.times[0] + 10*60+10 + idx*60*10# 10 seconds to avoid edge artifacts
                if offset < filt_file.times[-1]:
                    tiempos.append([onset, offset])
                    good_segments.append(idx)
                else:
                    continue
       
        else:
            print('multiple bad segments')
            # impair numbers index good segments (between end of bad segment and beginning of good segment)
            for x in index_len_bad_good_segments:
                remainder = x%2
                if remainder != 0 or x ==0: 
                    good_segments.append(x)
                    
            
            for idx,good_index in enumerate(good_segments): 
                plus_minutes = all_[good_index]+ 10*60-10
                onset = all_[good_index]+10
                tiempos.append([onset, plus_minutes])
        
                while plus_minutes + 10*60+1 < all_[good_index+1]-10:
                    onset = plus_minutes+1
                    plus_minutes = plus_minutes + 10*60+1
                    tiempos.append([onset, plus_minutes])
                   
    
        zero_sample_R = int(0.3*filt_file.info['sfreq']) 
        sample_150 = int(0.45*filt_file.info['sfreq']) 
        sample_300 = int(0.6*filt_file.info['sfreq'])
        
    
        
        if good_segments:
            
            n_good_segments = np.arange(0,len(tiempos)).tolist()
            
            print('\x1b[1;35m'+'found ' + str(len(n_good_segments)) + ' good segment(s)'+'\x1b[1;35m')
            print("\x1b[0m")   
            
            keys = ['R_Peaks_all','R_Peaks_clean','manual_rej_R_Peaks','R_Peaks_clean_events','ecg_data_good','tmin','tmax']#*len(good_segments)
          
            key_dict = {}
            for key in keys:
                key_dict[key] = []
                nested_dict = {}
            for dictionary in n_good_segments:
                nested_dict[dictionary] = copy.deepcopy(key_dict)
                
            subject_info['R_info'] = nested_dict    
   
            for idx,good_time in enumerate(tiempos): 
                     
                good_data = filt_file.copy().crop(tmin = good_time[0],tmax= good_time[1]) # get chunk of good data
                ECG = good_data.get_data(picks=filt_file.info.ch_names[-2:], start=0, stop=None, 
                                                         reject_by_annotation='NaN', return_times=False) # get ECG channel data
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
                print('\x1b[1;35m'+'Detecting R Peaks for segment ' + str(idx) +'\x1b[1;35m')
                print("\x1b[0m")   
              
                ECG_diff_clean = nk.ecg_clean(ECG_diff,sampling_rate=filt_file.info['sfreq'], method='neurokit')
                R_Peaks = nk.ecg_findpeaks(ECG_diff_clean,  sampling_rate = filt_file.info['sfreq'], method='neurokit', show=plot_R_detection)['ECG_R_Peaks']
                
                if plot_R_detection==True:
                    plt.show()
                    plt.pause(3)
                    plt.close()
                
                #create heartbeat epochs, plot and check if electrode substraction is correct. ECG channels seem to be located not consistently in left right 
                heartbeats = nk.epochs_create(ECG_diff_clean, R_Peaks, 
                                              sampling_rate= filt_file.info['sfreq'],
                                              epochs_start=-0.3,epochs_end=0.450,baseline_correction=True)
                
                heartbeats_plot = nk.epochs_to_df(heartbeats) # heartbeats to df
                heartbeats_plot['Label'] = heartbeats_plot['Label'].astype(int)
                heartbeats_pivoted = heartbeats_plot.pivot(index="Time", columns="Label", values="Signal")
                
                median_R_Peak_value = np.nanmedian(heartbeats_pivoted.loc[heartbeats_pivoted.index.values[zero_sample_R]].values) 
                median_150_300 = np.nanmedian(heartbeats_pivoted.loc[heartbeats_pivoted.index.values[sample_150:sample_300]].values)                                          
                print('median voltage value at 0 is ' + str(median_R_Peak_value)) 
                print('median voltage value at 150-300ms is  ' + str(median_150_300 )) 
                
                fig,ax = plt.subplots(1,1)
                
                cmap = iter(plt.cm.YlOrRd(np.linspace(0, 1, num=len(np.unique(heartbeats_pivoted.columns.values)))))  
             
                lines = []
                for x, color in zip(heartbeats_pivoted, cmap):  
                    (line,) = ax.plot(heartbeats_pivoted[x], color=color,label=x)
                    lines.append(line)
                mplcursors.cursor(lines, highlight=True)
                plt.xlabel("Time (s)")
                plt.title("Individual Heart Beats")                        
                # REMOVE HEARTBEATS WRONGLY DETECTED
                
                rpeaks_2_remove=[]
                rpeaks_2_remove_key =[]
                
                if file_id not in ['117_48677be7','597_3708d0ac','785_92289bbc', '791_62f722dc','452_2e2789b0','EBC024_a5850','EBC026_9951b','EBC028_0116f','EBC033_53fc6','EBC035_c485d','EBC045_3953c',
'EBC045_b6558', 'EBC045_ede7c']:
                    for heartbeat in heartbeats:
                        if median_150_300<0:
                            if(heartbeats[heartbeat]['Signal'].values[zero_sample_R]<abs(np.nanmean(heartbeats[heartbeat]['Signal'].values/4)) or 
            heartbeats[heartbeat]['Signal'].values[zero_sample_R]>abs(median_R_Peak_value)*3 or 
            np.nanmean(heartbeats[heartbeat]['Signal'].values[sample_150:sample_300])<median_150_300*6 or
            any(heartbeats[heartbeat]['Signal'].values[sample_150:sample_300]>abs(median_150_300)*10)):                          
                                r_peak_position = int(heartbeat)-1
                                rpeaks_2_remove_key.append(int(heartbeat))
                                rpeaks_2_remove.append(R_Peaks[r_peak_position]) 
                                
                                
                        elif median_150_300>0 and file_id not in ['EBC001_1e32c','EBC012_2387a','EBC012_837e1','EBC012_a5f90','EBC012_ab57c']:
                            if(heartbeats[heartbeat]['Signal'].values[zero_sample_R]<abs(np.nanmean(heartbeats[heartbeat]['Signal'].values/4)) or 
                heartbeats[heartbeat]['Signal'].values[zero_sample_R]>abs(median_R_Peak_value)*3 or 
                np.nanmean(heartbeats[heartbeat]['Signal'].values[sample_150:sample_300])<median_150_300/10 or
                any(heartbeats[heartbeat]['Signal'].values[sample_150:sample_300]>abs(median_150_300)*10)):                          
                                r_peak_position = int(heartbeat)-1
                                rpeaks_2_remove_key.append(int(heartbeat))
                                rpeaks_2_remove.append(R_Peaks[r_peak_position])
                        else:
                            if(heartbeats[heartbeat]['Signal'].values[zero_sample_R]<abs(np.nanmean(heartbeats[heartbeat]['Signal'].values/10)) or 
heartbeats[heartbeat]['Signal'].values[zero_sample_R]>abs(median_R_Peak_value)*3 or 
abs(np.nanmean(heartbeats[heartbeat]['Signal'].values[sample_150:sample_300]))<median_150_300/200 or
any(heartbeats[heartbeat]['Signal'].values[sample_150:sample_300]>abs(median_150_300)*10)):                          
                                r_peak_position = int(heartbeat)-1
                                rpeaks_2_remove_key.append(int(heartbeat))
                                rpeaks_2_remove.append(R_Peaks[r_peak_position]) 
                            
                                
                elif file_id in ['117_48677be7','597_3708d0ac','785_92289bbc', '791_62f722dc']:                
                    for heartbeat in heartbeats:
                        if(heartbeats[heartbeat]['Signal'].values[zero_sample_R]<abs(np.nanmean(heartbeats[heartbeat]['Signal'].values/4)) or 
           heartbeats[heartbeat]['Signal'].values[zero_sample_R]>abs(median_R_Peak_value)*3 or 
           abs(np.nanmean(heartbeats[heartbeat]['Signal'].values[sample_150:sample_300]))<median_150_300/15 or
           any(heartbeats[heartbeat]['Signal'].values[sample_150:sample_300]>abs(median_150_300)*100)):                         
                           r_peak_position = int(heartbeat)-1
                           rpeaks_2_remove_key.append(int(heartbeat))
                           rpeaks_2_remove.append(R_Peaks[r_peak_position])
                else:
                    for heartbeat in heartbeats:
                        if abs(heartbeats[heartbeat]['Signal'].values[zero_sample_R])<np.nanmean(heartbeats[heartbeat]['Signal'].values/6):                         
                            r_peak_position = int(heartbeat)-1
                            rpeaks_2_remove_key.append(int(heartbeat))
                            rpeaks_2_remove.append(R_Peaks[r_peak_position])                                        
                           
            
                        
                blues = iter(plt.cm.cool(np.linspace(0, 1, num=len(rpeaks_2_remove_key))))                        
                for x,color1 in zip(rpeaks_2_remove_key,blues):
                    (line,) = plt.plot(heartbeats_pivoted[x], color=color1)
                    lines.append(line)
                # plt.ylim([-3*(abs(median_R_Peak_value)),4*abs(median_R_Peak_value)])
    
                plt.show()
                plt.pause(4)
                plt.show()
                plt.close()
                
                fig,ax = plt.subplots(1,1)
                
                cmap = iter(plt.cm.YlOrRd(np.linspace(0, 1, num=len(np.unique(heartbeats_pivoted.columns.values)))))  
             
                lines = []
                for x, color in zip(heartbeats_pivoted, cmap):
                    if x not in rpeaks_2_remove_key:
                        (line,) = ax.plot(heartbeats_pivoted[x], color=color,label=x)
                        lines.append(line)
                mplcursors.cursor(lines, highlight=True)
                plt.xlabel("Time (s)")
                plt.title("Individual Heart Beats")
                print('close plot to continue')
                plt.show(block=True)    
                
                if visual_check == True:
                    manual_selec = input('REMOVE SPECIFIC HEARTBEATS (type numbers separated by space) ')
    
                    
                    if manual_selec  !='':
                        manual_selec = list(manual_selec.split(" ")) 
                        for hb in manual_selec:                       
                            rpeaks_2_remove.append(R_Peaks[int(hb)-1])             
                            rpeaks_2_remove_key.append(int(hb)) 
                     
                        
                #PLOT AGAIN TO CHECK FINAL REJECTION
                    fig,ax = plt.subplots(1,1)
                    cmap = iter(plt.cm.YlOrRd(np.linspace(0, 1, num=len(np.unique(heartbeats_pivoted.columns.values)))))  
                    
                    lines = []
                    for x, color in zip(heartbeats_pivoted, cmap):
                        if x not in rpeaks_2_remove_key:
                            (line,) = ax.plot(heartbeats_pivoted[x], color=color,label=x)
                            lines.append(line)
                    mplcursors.cursor(lines, highlight=True)
                    plt.show()
                    
                    # plt.ylim([-3*(abs(median_R_Peak_value)),4*abs(median_R_Peak_value)])
                    plt.xlabel("Time (s)")
                    plt.title("Individual Heart Beats")
                    plt.show()
                    plt.pause(2)
                    print('saving plot')
    
                    fig.savefig(join(cfg.images_path,file_id+'_'+str(idx)+'_ECG.pdf'),dpi=300)
                    plt.close()
                
                #remove wrongly detected R peaks
                rpeaks_2_remove.sort()
                R_Peaks_clean = R_Peaks.tolist() 
                
                for x in rpeaks_2_remove:
                    R_Peaks_clean.remove(x)  
                    
                R_Peaks_clean = np.array(R_Peaks_clean)
                   
                print('Rejected ' +str(len(rpeaks_2_remove)) + ' of '+str(len(R_Peaks)) +' heartbeats')
                    
                correct_R_Peaks = int((good_time[0])*filt_file.info['sfreq'])
                trig_code = np.repeat(1,len(R_Peaks_clean))
                R_Peaks_events = np.stack([R_Peaks_clean + correct_R_Peaks ,R_Peaks_clean+correct_R_Peaks+1, trig_code], axis=1) # 
                
                subject_info['R_info'][idx]['R_Peaks_all'] =  R_Peaks
                subject_info['R_info'][idx]['R_Peaks_clean'] =  R_Peaks_clean
                subject_info['R_info'][idx]['manual_rej_R_Peaks'] =  manual_selec
                subject_info['R_info'][idx]['R_Peaks_clean_events'] = R_Peaks_events
                
                data_eval = input('ECG data good? (y-n-yn): ')
                subject_info['R_info'][idx]['ecg_data_good'] = data_eval           
                subject_info['R_info'][idx]['tmin'] = good_time[0]
                subject_info['R_info'][idx]['tmax'] = good_time[1]
                
                onsets.append(subject_info['R_info'][idx]['tmin'])
                offsets.append(subject_info['R_info'][idx]['tmax'])
                durations.append(subject_info['R_info'][idx]['tmax']-subject_info['R_info'][idx]['tmin'])
                description.append('good')
                    
            good_annotations = mne.Annotations(onset=onsets,duration = durations,description = description,orig_time=filt_file.annotations.orig_time)  
                
            filt_file.set_annotations(filt_file.annotations + good_annotations)
            subject_info['annotations'] = filt_file.annotations 

            # subject_info['annotations'] = filt_file.annotations + good_annotations
            
            filt_file.plot(block = True, duration = 150,n_channels =22,scalings =dict(eeg=100e-6,ecg=5e-4))
            data_eval = input('EEG data good? (y-n-yn): ') 
            subject_info['eeg_data_good'] = data_eval
            raw_w_good_annot_name = file_id +'_' + str(filt_file.info['highpass']) + '_' + str(filt_file.info['lowpass'])+'_60Hz_notch' +'_annot_good_raw.fif'
            raw_file_path = join(cfg.annot_files_path,raw_w_good_annot_name)
            filt_file.save(raw_file_path, overwrite=True)
        else:
            print('no good segments!!!')
            print(file_id)
            print('file_index: ' +str(file_index))
            sys.exit()        
        pickle.dump(subject_info,open(subject_info_dict_path, "wb"))
    else:
        raw_w_good_annot_name = file_id +'_' + str(filt_file.info['highpass']) + '_' + str(filt_file.info['lowpass'])+'_60Hz_notch' +'_annot_good_raw.fif'
        raw_file_path = join(cfg.annot_files_path,raw_w_good_annot_name)
    
        filt_file.save(raw_file_path, overwrite=True)
        print('file already processed')

    return subject_info

    
        

#========================================================================
#
# EPOCH DATA AND CLEAN
# - epoch data to HEP or 2s epochs to calculate markers
# - noisy epochs and channels are rejected or interpolated using autoreject
# - running this function when -epo file already exists allos to visualize
# the average of epochs and manually select noisy channel that wasn't detected
# by autoreject and is interpolated
#=========================================================================        

def epoch_data_n_clean_autoreject(file,file_id,epoch_type,cfg,manual_ch_rejection,overwrite):
    
    file_path_name = join(cfg.annot_files_path,file)

    subject_info_dict_path = join(cfg.info_path,file_id+'_info.pkl')
    subject_info_open = open(subject_info_dict_path , "rb")
    subject_info = pickle.load(subject_info_open)  

    baseline = cfg.epoch_param[epoch_type]['baseline']
    tmin = cfg.epoch_param[epoch_type]['tmin']
    tmax = cfg.epoch_param[epoch_type]['tmax']
    detrend = cfg.epoch_param[epoch_type]['detrend']
    
    n_bads_threshold =  cfg.epoch_param[epoch_type]['n_bads_threshold']
    n_interpolates = cfg.epoch_param[epoch_type]['n_interpolates'] 
    consensus_percs = cfg.epoch_param[epoch_type]['consensus_percs'] 
    
    filtered_file = mne.io.read_raw_fif(file_path_name, preload = True,verbose='ERROR')
    filtered_file.set_montage('standard_1020',match_case=True,verbose=True )
    filtered_file.drop_channels(filtered_file.info.ch_names[-2:])
    sf = filtered_file.info['sfreq']         
    
    if 'R_info' in list(subject_info.keys()):
        
        for chunk_ok_data in list(subject_info['R_info'].keys()):
            print(chunk_ok_data)
            
            if epoch_type =='HEP':
                
                epochs_path = cfg.epochs_path_HEP
            
                if (subject_info['eeg_data_good']=='y' or subject_info['eeg_data_good']=='yn') and (subject_info['R_info'][chunk_ok_data]['ecg_data_good'] =='y' or subject_info['R_info'][chunk_ok_data]['ecg_data_good'] =='yn'):
                    events = subject_info['R_info'][chunk_ok_data]['R_Peaks_clean_events'] # only use good epochs               
                    epochs_params = dict(events=events, tmin=tmin, tmax=tmax,
                                 baseline=baseline, detrend = detrend, reject_by_annotation=True)
                    print('epoching segment' + str(chunk_ok_data) +' from file '+ file_id)
                    epochs = mne.Epochs(filtered_file, **epochs_params, preload=True)
                    
                else:
                    print('no ecg events to epoch data')
                    continue

            if epoch_type == 'markers':
                
                epochs_path = cfg.epochs_path_markers
        
                if subject_info['eeg_data_good']=='y'  or subject_info['eeg_data_good']=='yn':


                    tmin_segment = subject_info['R_info'][chunk_ok_data]['tmin']
                    tmax_segment = subject_info['R_info'][chunk_ok_data]['tmax']
                    good_data = filtered_file.copy().crop(tmin=tmin_segment,tmax=tmax_segment)
                               
                    len_new_epochs = int(tmax-tmin)
                    n_samples = int(sf*len_new_epochs)
                    
                    new_events = list()
                    outcome_info = list()
                    gose_dc_info = list()
                    command_score_dc_info = list()
                    segment_info = list()
                    
                    for repeat in range(cfg.epoch_param[epoch_type]['n_epo_segments']):
                        event_corrected = [repeat*n_samples +tmin_segment*sf,0,999] # define event info for new short epoch [on sample epoch, 0, trig code]
                        new_events.append(event_corrected) # append event to all new events
                        outcome_info.append(subject_info['outcome']) 
                        gose_dc_info.append(subject_info['gose_dc']) 
                        if  'command_score_dc' in subject_info:
                            command_score_dc_info.append(subject_info['command_score_dc'])
                        elif 'command_last' in subject_info:
                            command_score_dc_info.append(subject_info['command_last'])
                        segment_info.append(chunk_ok_data)

   
                    new_events = np.array(new_events,int)
                    
                    info_dict = dict(time_sample = new_events[:,0], 
                                              segment=segment_info,
                                              outcome=outcome_info,
                                              gose_dc = gose_dc_info,
                                              command_score_dc = command_score_dc_info)
                   
                    metadata = pd.DataFrame.from_dict(info_dict)

    
                    epochs_params = dict(events=new_events, tmin=tmin, tmax=tmax,
                                          baseline=baseline, detrend = detrend, reject_by_annotation=True,
                                          metadata = metadata)
                    print('re epoching file '+file)   
                    
                    epochs = mne.Epochs(good_data, **epochs_params, preload=True)
                else:
                    print('eeg signal was set as bad')
                    continue

                  
                       
            epochs_file_name =  file_id+'_'+ str(chunk_ok_data) +'-epo.fif'
            epochs_file_path = join(epochs_path,epochs_file_name) 
            
            if not isfile(epochs_file_path):
                ar = AutoReject([0], [1.0],n_jobs=-1,random_state=42)
                ar.fit(epochs)
                reject_log = ar.get_reject_log(epochs)
                                
                n_epochs = len(epochs)
                n_bads = reject_log.labels.sum(axis=0)
                bad_chs_idx = np.where(n_bads > n_epochs * n_bads_threshold)[0]
                bad_chs = [epochs.ch_names[idx] for idx in bad_chs_idx]
                print('globally bad according to Autoreject: ')
                print(bad_chs)
                
                epochs.info['bads'] = bad_chs            
                
                ar = AutoReject(n_interpolates, consensus_percs,
                        thresh_method='bayesian_optimization', random_state=42, n_jobs=-1)
                ar.fit(epochs)
                
                epochs_clean, reject_log = ar.transform(epochs, return_log=True)  
                epochs_clean.interpolate_bads(reset_bads=True)
                epochs_clean.set_eeg_reference(ref_channels='average')
                
                if epochs_clean.info['sfreq']!=256:
                    epochs_clean.resample(256,npad='auto',window='boxcar',n_jobs=1,pad='edge',verbose=None)
    
                epochs_clean.save(epochs_file_path,overwrite=True, verbose=True)
                
                # To create .csv report of bad epochs, bad ch, and interpolated ch
                bad_epochs = np.where(reject_log.bad_epochs)[0]
                np.savetxt(join(epochs_path,epochs_file_name[:-8]+'_rejected_epochs.csv'), bad_epochs, delimiter=",")
                interp_trial_ch = np.asarray(np.where(reject_log.labels==2)).T.tolist()
                ch_names = epochs_clean.info.ch_names
                                       
                for idx,trial_ch in enumerate(interp_trial_ch):   
                    interp_trial_ch[idx][1] = ch_names[trial_ch[1]]
                    
                interp_trial_ch_df = pd.DataFrame(interp_trial_ch,columns=['trial','interpolated channel'])    
                interp_trial_ch_df.to_csv(join(epochs_path,epochs_file_name[:-8]+'_interp_channels_per_epoch.csv'),index=False) 
    
        
            elif manual_ch_rejection == True:
                
                epochs = mne.read_epochs(epochs_file_path)

                evoked = epochs.average()
                
                fig1, ax1 = plt.subplots(1,1)
                fig1.suptitle(file[:8])
                evoked.plot(axes=ax1,spatial_colors=True,gfp=True)
        
                plt.show(block=True)
                
                manual_selec = input('\n Channels to interpolate (separated by comma): ')
    
                    
                if manual_selec  !='':
                    manual_selec_df = list(manual_selec.split(",")) 
                    df = pd.DataFrame(manual_selec_df)
                    df.to_csv(join(epochs_path,epochs_file_name[:-8]+'_manual_rej_channels.csv'), index=False,header=False)

                    epochs.info['bads'] = manual_selec_df   
                    epochs.interpolate_bads(reset_bads=True)
                    epochs.save(epochs_file_path,overwrite=True, verbose=True)
                    
                    evoked = epochs.average()
                    
                    fig1, ax1 = plt.subplots(1,1)
                    fig1.suptitle(file[:8])
                    evoked.plot(axes=ax1,spatial_colors=True,gfp=True)
            
                    plt.show()
                    plt.pause(2)
                    
                eeg_segment_cat = input('\n good eeg segment? (y/n/yn): ')
                
                if epoch_type=='markers':
                    subject_info['R_info'][chunk_ok_data]['eeg_data_good']= eeg_segment_cat
                if epoch_type =='HEP':
                    subject_info['R_info'][chunk_ok_data]['HEP_data_good']= eeg_segment_cat

                pickle.dump(subject_info,open(subject_info_dict_path, "wb"))
   
                    
        
                print("\x1b[0m")   
                plt.close('all')
                
                
                                      