#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Em-Fl (emilia.flo.rama at gmail)
"""
import mne
import matplotlib.pyplot as plt
from os.path import join, isfile
import pandas as pd
import numpy as np
import pickle
from matplotlib.ticker import FormatStrFormatter
from datetime import date
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import mplcursors

#========================================================================
#
# PLOT EVOKED ERP WITH CI
# blocks execution until evoked plot is closed. Allows to check lines plotted
# manually by using the cursos (mpl)
#=========================================================================  
def plot_compare_evokeds(evoked_data_all,epochs_included,n_epochs,group_by,conditions,channel,cfg,trace_plot,overwrite): 
    today = str(date.today())
    
    ERP_CI_fig = 'ERP_CI_' +group_by + '_'+ channel +'_'+today+'.pdf'
    ERP_CI_fig_path = join(cfg.images_path,ERP_CI_fig)
        
    styles = {conditions[0]:{"linewidth": cfg.linewidth},conditions[1]:{"linewidth": cfg.linewidth}}
    df = pd.DataFrame({'epochs_file':epochs_included,'number_epochs':n_epochs})
    files1 = df[df['epochs_file'].str.contains(conditions[0])]['epochs_file']
    files2 = df[df['epochs_file'].str.contains(conditions[1])]['epochs_file']
    
    
    if overwrite or not isfile(ERP_CI_fig_path):           
        
        fig,ax = plt.subplots()
        mne.viz.plot_compare_evokeds(evoked_data_all,picks=channel,ci=True,
                                     colors=[cfg.palette[0],cfg.palette[1]],
                                     axes=ax,styles=styles)
    

        plt.show()
        plt.pause(2)
        fig.savefig(ERP_CI_fig_path, dpi=300)
        plt.close()
        
        ERP_trace_fig = 'ERP_trace_' + group_by + '_'+channel + '_'+today+'.pdf'
        ERP_trace_fig_path = join(cfg.images_path,ERP_trace_fig)
        
                      
        if trace_plot == 'yes':
            
            cmap1 = iter(plt.cm.Blues(np.linspace(0, 1, num=len(evoked_data_all[conditions[0]]))))
            cmap2 = iter(plt.cm.Purples(np.linspace(0, 1, num=len(evoked_data_all[conditions[1]]))))
         
            lines1 = []
            lines2 = []
            
            fig,ax = plt.subplots(1,2) 
                        
            for (evoked1, color1,file_name) in zip(evoked_data_all[conditions[0]], cmap1,files1):
                evoked_1_df = evoked1.to_data_frame(picks=channel)
                (line,)=ax[0].plot(evoked_1_df['time'],evoked_1_df[channel],color=color1,label=file_name)
                lines1.append(line)
            mplcursors.cursor(lines1, highlight=True)
            
            
            for (evoked2, color2,file_name) in zip(evoked_data_all[conditions[1]], cmap2,files2):
                evoked_2_df = evoked2.to_data_frame(picks=channel)
                (line,)=ax[1].plot(evoked_2_df['time'],evoked_2_df[channel],color=color2,label=file_name)
                lines2.append(line)
            mplcursors.cursor(lines2, highlight=True)

            plt.show()

                
            GA_1 = mne.grand_average(evoked_data_all[conditions[0]]).to_data_frame(picks=channel)
            GA_2 = mne.grand_average(evoked_data_all[conditions[1]]).to_data_frame(picks=channel)
            
            ax[0].plot(GA_1['time'],GA_1[channel],color=cfg.palette[0],linewidth=cfg.linewidth)
            ax[1].plot(GA_2['time'],GA_2[channel],color=cfg.palette[1],linewidth=cfg.linewidth)
            
            y_min_A, y_max_A = ax[0].get_ylim()
            y_min_B, y_max_B = ax[1].get_ylim()  
            
            ymin = min(y_min_A,y_min_B)
            ymax = max(y_max_A,y_max_B)

            ax[0].set_ylim([ymin,ymax])
            ax[1].set_ylim([ymin,ymax])
            
            fig.savefig(ERP_trace_fig_path, dpi=300)
            plt.show(block=True)

         

#========================================================================
#
# PLOT TOPOGRAPHIES
# - plots difference between grouping categories
# - plots all patients together
#=========================================================================  
def plot_compare_topography(evoked_data_all,group_by,conditions,cfg,overwrite): 
    today = str(date.today())

    topo_diff_fig = group_by + '_topo_diff' +'_'+today+'.pdf'
    topo_diff_fig_path = join(cfg.images_path,topo_diff_fig)

    if overwrite or not isfile(topo_diff_fig_path):
             
        GA_1 = mne.grand_average(evoked_data_all[conditions[0]])
        GA_2 = mne.grand_average(evoked_data_all[conditions[1]])

        all_times = np.arange(0.1, GA_1.times[-1], 0.2)

        evoked_diff = mne.combine_evoked([GA_1, GA_2],
                            weights=[1,-1])  # calculate difference wave
            
      
        evoked_diff.plot_topomap(all_times, ch_type='eeg', time_unit='s',ncols=8, nrows='auto')                                                  
                                                  

        

        plt.savefig(topo_diff_fig_path, dpi=300)
    
        plt.close('all')
        all_ = [GA_1,GA_2]
        GA_all = mne.grand_average(all_)
        all_times = np.arange(0.1, GA_1.times[-1], 0.1)

        GA_all.plot_topomap(all_times, ch_type='eeg', time_unit='s',ncols=8, nrows='auto')                                                  
        topo_all_fig = group_by + '_topo_GA_all' +'_'+today+'.pdf'
        topo_all_fig_path = join(cfg.images_path,topo_all_fig)
        plt.show()
        plt.pause(2)
        plt.savefig(topo_all_fig_path, dpi=300)
        plt.close('all')




 #========================================================================
 #
 # PLOT AVERAGE ELECTRODES THAT TAKE PART IN SIGNIFICANT CLUSTER
 # If a significant cluster is found plot electrodes that are included and 
 # with time window of effect
 #=========================================================================             
def plot_significant_electrodes(evoked_data_all,epoch_type,group_by,cfg,cluster_info):
     
    conditions = list(evoked_data_all.keys())
    cluster_info_open = open(join(cfg.statistics_path,cluster_info), "rb")
    cluster_dict = pickle.load(cluster_info_open)  
    cluster_p_values = cluster_dict['cluster_p_values']
    clusters = cluster_dict['clusters']
    
    sig_clusters = np.where(cluster_p_values<cfg.cluster_p_value)[0]
    
    if len(sig_clusters)>0:
        sf = evoked_data_all[conditions[0]][0].info['sfreq']
        for cluster in sig_clusters:
            
            elec_in_cluster_fig  = cluster_info[:-8]+ '_elecs_in_cluster_' +str(cluster) +'.pdf'
            elec_in_cluster_fig_path = join(cfg.images_path,elec_in_cluster_fig)
    
            picks=[]
            
            electrodes = np.unique(np.where(clusters[cluster])[1]) # get electrodes
            electrode_names = []
            samples = np.unique(np.where(clusters[cluster])[0]) # get time points
            t0 = samples[0]
            tf = samples[-1]
            
            if cluster_dict['time_window'][0] != round(evoked_data_all[conditions[0]][0].times[0],2):
                t0 = t0 + round(abs(round(evoked_data_all[conditions[0]][0].times[0],2))*sf)
                tf = tf + round(abs(round(evoked_data_all[conditions[0]][0].times[0],2))*sf)
    
               
            for e in electrodes:
                electrode_names.append(evoked_data_all[conditions[0]][0].info['ch_names'][e])
                picks.append(evoked_data_all[conditions[0]][0].info['ch_names'][e])
         
            v1 = round((t0/sf)+evoked_data_all[conditions[0]][0].tmin,3)
            
            v2 =  round((tf/sf)+evoked_data_all[conditions[0]][0].tmin,3)
                
            fig,ax = plt.subplots()
        
            
            mne.viz.plot_compare_evokeds(evoked_data_all,picks=picks,ci=True,
                                         colors=[cfg.palette[0],cfg.palette[1]],
                                         styles={conditions[0]: {"linewidth": cfg.linewidth},conditions[1]: {"linewidth": cfg.linewidth}},
                                          axes=ax,vlines='auto',combine='mean',title='',truncate_xaxis=False,truncate_yaxis=False,
                                         show_sensors=False)
                
            
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            leg = ax.legend()
            
            for line in leg.get_lines():
                line.set_linewidth(4.0)
    
            plt.legend(frameon=False, loc='lower right')
            
            plt.axvspan(v1, v2, facecolor='silver', alpha=0.5)
            axins = inset_axes(ax, "25%", "25%")
            axins.set_frame_on(False)
            axins.set_xticks([])
            axins.set_yticks([])
            evoked_sensors = evoked_data_all[conditions[0]][0].copy().pick(electrode_names)
        
            mne.viz.plot_sensors(evoked_sensors.info, kind='topomap', axes=axins,ch_type=None, title=None, show_names=False, ch_groups=None, to_sphere=True,block=False, show=True, sphere=None, pointsize=3.5, linewidth=2, verbose=None)
       
            plt.show()
    
            plt.pause(2)
            fig.savefig(elec_in_cluster_fig_path ,  dpi=300)
            plt.close()
    else:
        print('\n no significant clusters were found for this contrast \n')

              
                             

