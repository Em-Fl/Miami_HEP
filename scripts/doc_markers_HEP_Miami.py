      # -*- coding: utf-8 -*-
"""
@author: Em-Fl (emilia.flo.rama at gmail)

modified from @DraganaMana !

"""
import os
import os.path as op
from os.path import join, isfile, isdir

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
import time
import os
import mne
import ptitprince as pt
import nice 
from nice.markers import (KolmogorovComplexity, PowerSpectralDensityEstimator, 
                          PowerSpectralDensity, SymbolicMutualInformation, PermutationEntropy)

from sklearn.metrics import roc_auc_score
import scipy
from autoreject import AutoReject, get_rejection_threshold, compute_thresholds
import pickle
from scipy.stats import mannwhitneyu
import seaborn as sns

#========================================================================
#
# EEG MARKERS:
#


# KolmogorovComplexity/default
# PermutationEntropy/default by default is for THETA
# SymbolicMutualInformation/weighted by default if for THETA
       
# PowerSpectralDensity/delta
# PowerSpectralDensity/deltan
       
# PowerSpectralDensity/theta
# PowerSpectralDensity/thetan
       
       
# PowerSpectralDensity/alpha
# PowerSpectralDensity/alphan
       
# PowerSpectralDensity/beta
# PowerSpectralDensity/betan

# PowerSpectralDensity/gamma
# PowerSpectralDensity/gamman

#IMPORTAAAANT SO IT RUNS

# runs on mne 0.24
# nice has to be installed using: python setup.py develop 
# in environment of interest
#Then set backend='c' or 'openmp' instead of the defalut backend='python' in markers functions



#=========================================================================  



def get_eeg_markers(file,files,tmin,tmax, df_all_subj,df_epochs_all_subj,cfg): 
    
    print('\n getting eeg markers for \n')
    print(file)
    
    chunk_ok_data = file.split('_')[2][:-8]
    subject_info_open = open(join(cfg.info_path,file[0:12]+'_info.pkl' ), "rb")
    subject_info = pickle.load(subject_info_open)    

    epochs = mne.read_epochs(join(cfg.epochs_path_markers,file))


    elecs_extra = ['T3','T4','T5','T6']
    for e in elecs_extra:
        if e in epochs.info.ch_names:
            epochs = epochs.drop_channels(e)
            print('\n dropping channels \n')
    
    epochs_new_index = epochs.metadata.reset_index()
    n_epochs_for_markers = len(epochs)
    eeg_segment_good= subject_info['R_info'][int(chunk_ok_data)]['eeg_data_good']
    ecg_segment_good= subject_info['R_info'][int(chunk_ok_data)]['ecg_data_good']
    tmin_s = subject_info['R_info'][int(chunk_ok_data)]['tmin']
    tmax_s = subject_info['R_info'][int(chunk_ok_data)]['tmax']

    outcome =  epochs_new_index['outcome'][0]
    command_score_dc = epochs_new_index['command_score_dc'][0]
    gose_dc =  epochs_new_index['gose_dc'][0]

   
    #############################################################################
    #########################################SPECTRAL############################
    
      #PowerSpectralDensityL
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto', nperseg=128)
    base_psd = PowerSpectralDensityEstimator(
        psd_method='welch', tmin=tmin, tmax=tmax, fmin=1., fmax=40.,
        psd_params=psds_params, comment='default')

    reduction_func = [
        {'axis': 'epochs', 'function': np.mean},
        {'axis': 'channels', 'function': np.mean},
        {'axis': 'frequency', 'function': np.sum}]

    #alpha normalized
    alpha_n = PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=13.,normalize=True, comment='alphan')
    alpha_n.fit(epochs)
    dataalpha_n = alpha_n._reduce_to(reduction_func, target='epochs', picks=None)


    #alpha
    alpha = PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=13.,normalize=False, comment='alpha')
    alpha.fit(epochs)
    dataalpha = alpha._reduce_to(reduction_func, target='epochs', picks=None)

    #delta normalized
    delta = PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4., normalize=True, comment='delta')
    delta.fit(epochs)
    datadelta_n = delta._reduce_to(reduction_func, target='epochs', picks=None)

    #delta
    delta = PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4, normalize=False, comment='delta')
    delta.fit(epochs)
    datadelta = delta._reduce_to(reduction_func, target='epochs', picks=None)

    #theta normalized
    theta = PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8., normalize=True, comment='theta')
    theta.fit(epochs)
    datatheta_n = theta._reduce_to(reduction_func, target='epochs', picks=None)

    #theta
    theta = PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8,normalize=False, comment='theta')
    theta.fit(epochs)
    datatheta = theta._reduce_to(reduction_func, target='epochs', picks=None)

    # #gamma normalized
    # gamma = PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,normalize=True, comment='gamma')
    # gamma.fit(epochs)
    # datagamma_n = gamma._reduce_to(reduction_func, target='epochs', picks=None)

    # #gamma
    # gamma = PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45,normalize=False, comment='theta')
    # gamma.fit(epochs)
    # datagamma = gamma._reduce_to(reduction_func, target='epochs', picks=None)

    #beta normalized
    beta = PowerSpectralDensity(estimator=base_psd, fmin=13., fmax=30.,normalize=True, comment='beta')
    beta.fit(epochs)
    databeta_n = beta._reduce_to(reduction_func, target='epochs', picks=None)

    #beta
    beta = PowerSpectralDensity(estimator=base_psd, fmin=13., fmax=30,normalize=False, comment='beta')
    beta.fit(epochs)
    databeta = beta._reduce_to(reduction_func, target='epochs', picks=None)


    ###########################################################################################
    ######################################### INFORMATION THEORY ##############################

    komplexity = KolmogorovComplexity(tmin=tmin, tmax=tmax, backend='openmp')
    komplexity.fit(epochs)
    # komplexityobject=komplexity.data_ ###Object to save, number of channels*number of epochs, it's ndarray

    reduction_func = [
    {'axis': 'epochs', 'function': np.mean},
    {'axis': 'channels', 'function': np.mean}]

    datakomplexity = komplexity._reduce_to(reduction_func, target='epochs', picks=None)


    p_e = PermutationEntropy(tmin=tmin, tmax=tmax)
    p_e.fit(epochs)
    # p_eobject = p_e.data_
    datap_e = p_e._reduce_to(reduction_func, target='epochs', picks=None)

    ###########################################################################################
    ######################################### wSMI ############################################
    
    wSMI = SymbolicMutualInformation(tmin=tmin, tmax=tmax, #kernel=3, tau=8, 
                                     backend="openmp",
                  method_params=None, method='weighted', comment='default')
    wSMI.fit(epochs)
    # wSMIobject = wSMI.data_

    reduction_func = [
    {'axis': 'epochs', 'function': np.mean},
    {'axis': 'channels', 'function': np.mean},
    {'axis': 'channels_y','function':np.mean}]

    datawSMI = wSMI._reduce_to(reduction_func, target='epochs', picks=None)
    # Add all values to a dataframe
    
    
    df = pd.DataFrame([[file[0:12],chunk_ok_data,tmin_s,tmax_s,n_epochs_for_markers,eeg_segment_good,ecg_segment_good,outcome,gose_dc,command_score_dc,np.mean(datakomplexity), np.mean(datap_e), np.mean(datawSMI), 
                          np.mean(datadelta), np.mean(datadelta_n), np.mean(datatheta), np.mean(datatheta_n),
                          np.mean(dataalpha), np.mean(dataalpha_n), np.mean(databeta), np.mean(databeta_n)]], 
                       columns = cfg.column_names_markers)
  
    
    df_all_subj = pd.concat([df_all_subj,df])
        
    df_epochs = pd.DataFrame([[file[0:12],chunk_ok_data,tmin_s,tmax_s,n_epochs_for_markers,eeg_segment_good,ecg_segment_good,outcome,gose_dc,command_score_dc,datakomplexity, datap_e, datawSMI, 
                      datadelta, datadelta_n, datatheta, datatheta_n,
                      dataalpha, dataalpha_n, databeta, databeta_n]],
                       columns = cfg.column_names_markers) # checking sth
        
    df_epochs_all_subj = pd.concat([df_epochs_all_subj,df_epochs])
    
        
    df_epochs_all_subj = df_epochs_all_subj.reset_index()
    df_epochs_all_subj.drop(labels=['index'],axis=1,inplace=True)
    df_epochs_all_subj.to_pickle(join(cfg.markers_path,'df_markers_epochs_all_subj.pkl'),protocol=4)  
    
    df_all_subj = df_all_subj.reset_index()
    df_all_subj.drop(labels=['index'],axis=1,inplace=True)
    df_all_subj.to_pickle(join(cfg.markers_path,'df_markers_all_subj.pkl'),protocol=4)  
      
    # Append the subject df to the wholesome df
    return df_all_subj,  df_epochs_all_subj

    
        