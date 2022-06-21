from dis import dis
import os
import json
import pandas as pd
import glob
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from functions.helper import normJerk, readTaskCSV,getIndexNames,getMarkerlessData,distance3D,pathAccelJerk
from functions.SessionOutputData import SessionOutputData

class SessionDataObject:
    def __init__(self, path: str,plot: bool,height: float,walking=True,ng=True):
        self.id: str = path[-14:-11]
        self.path: str = path
        self.date: str = path[-10:]
        self.markerless_data: dict[str:pd.DataFrame]
        self.markerless_events: dict[str:pd.DataFrame]
        self.markerless_fs: int = 85
        self.markerless_task_labels: pd.DataFrame
        self.marker_data: dict[str:pd.DataFrame]
        self.marker_fs: int = 100
        self.marker_task_labels: pd.DataFrame
        self.height = height

        self.markerless_output_data: pd.DataFrame
        self.marker_output_data: pd.DataFrame

        self.plot = plot

        self.output_data = SessionOutputData(self.id)
        
        # this path will be the path to each subject's folder and the date format for their session
        # for instance "...\\002\\2022-01-28"
        self.s_print("loading data for date %s" %  path[-10:])

        # this uses the fact that all markerless will be 001 and all marker will be 002
        if os.path.isdir(self.path + '_001'):
            self.load_markerless()
            if any(getIndexNames('Walking',self.markerless_task_labels)):
                self.s_print('Walking')
                self.analyze_walking(walking)
            else:
                self.s_print("No walking trials")
            if any(getIndexNames('Tandem',self.markerless_task_labels)):
                self.s_print('Tandem')
                self.analyze_tandem(walking)
            else:
                self.s_print("No Tandem trials")
        else:
            self.s_print("Could not find markerless session for %s"% self.id)


        if os.path.isdir(self.path + '_002'):
            self.load_markers()
            if any(getIndexNames('NG',self.marker_task_labels)):
                self.s_print("NG")
                self.analyze_NG(ng)
            else:
                self.s_print("No NG trials")
            if any(getIndexNames('9HP',self.marker_task_labels)): 
                self.s_print("9HP")
                self.analyze_9HP(ng)
            else:
                self.s_print("No 9HP trials")
        else:
            self.s_print("Could not find marker session for %s"% self.id)


    def s_print(self,str):
        print('['+self.id+'] '+str)

    def getOutput(self, task_name : str, metric : str) -> np.ndarray:
        #TODO: Make this return none if does not exist
        if self.output_data.data[task_name][metric][0].shape == ():
            #this is for outputs of length 1
            return np.array(self.output_data.data[task_name][metric])
        return np.concatenate(self.output_data.data[task_name][metric])
        # return self.output_data.data[task_name][metric][-1]

    def load_markerless(self):
        self.markerless_data = {}
        self.markerless_events = {}
        self.markerless_data_unused = {}       
        with open(glob.glob(self.path+'_001'+'/*.json')[0]) as f:
            markerless_data_json = json.load(f)
        
        names_to_exclude = ['Head_wrt_Lab'] #TODO: this is a bit of a coarse workaround...but alot of numpy stuff doesnt work if there are things with na values floating around....

        for i in range(len(markerless_data_json['Visual3D'])):
            data = markerless_data_json['Visual3D'][i]
            trial_name = data['filename']
            trial_name = trial_name.split('/')[-1][0:-4]
            if trial_name not in self.markerless_data.keys():
                self.markerless_data[trial_name] = pd.DataFrame()
                self.markerless_events[trial_name] = {}
                self.markerless_data_unused[trial_name] = pd.DataFrame()
                #print(trial_name)
            for j in range(len(data['signal'])):
                if data['name'] in ['LHS','LTO','RHS','RTO']:
                    self.markerless_events[trial_name].update({data['name']:data['signal'][j]['data']})
                elif data['name'] in names_to_exclude:
                    self.markerless_data_unused[trial_name].update({data['name']:data['signal'][j]['data']})
                else:
                    self.markerless_data[trial_name][data['name']+data['signal'][j]['component']] = data['signal'][j]['data']
        
        for i in self.markerless_data.keys():
            self.markerless_data[i].dropna(how='all',inplace=True)

        self.markerless_task_labels = readTaskCSV(self.path+'_001')
        self.markerless_output_data = self.markerless_task_labels

        
    def load_markers(self):
        marker_files = glob.glob(self.path + '_002' + '/*.tsv')
        self.marker_data = {}
        for i in marker_files:
            name = (i.split('\\')[-1]).split('.')[0]
            data = pd.read_csv(i,sep='\t',skiprows=11)
            self.marker_data[name] = data
            #print(name)
        self.marker_task_labels = readTaskCSV(self.path+'_002')
        self.marker_output_data = self.marker_task_labels


    def analyze_walking(self,plot):
        self.walking_angle(plot)
        self.calculate_step_width(plot)
        self.calculate_step_height(plot)
        self.calculate_pelvis_jerk_step(plot)
        self.calculate_thorax_jerk_step(plot)
        self.calculate_support(plot)
        self.calculate_joint_angles_walking(plot)
    
    def analyze_tandem(self,plot):
        self.calculate_pelvis_jerk_tandem(plot)
        self.calculate_thorax_jerk_tandem(plot)
        self.calculate_support_tandem(plot)
        
    def analyze_9HP(self,plot):
        if self.id == '002':
            return
        hp_keys = getIndexNames('9HP',self.marker_task_labels)
        if plot:
            fig_right,axs_right = plt.subplots(2,2)
            fig_left,axs_left = plt.subplots(2,2)
        for trial in hp_keys:

            # LOAD DATA
            data = self.marker_data[trial]
            data = data * 10**-3   # convert from mm to m

            if 'R' in data.columns[8][0] or 'L' in data.columns[8][0]:
                # remove the Right/Left from data. this is for the old data labeling method
                new_names = []
                for colname in data.columns:
                    if colname[0] == 'R':
                        colname = colname[1:]
                        right = True
                        # if right side, then the first peak will be positivie
                    elif colname[0] == 'L':
                        colname = colname[1:]
                        right = False
                    new_names.append(colname)
                data.columns = new_names

                if 'right' not in locals():
                    return
            else:
                #for new data labelling method
                match self.marker_task_labels.loc[trial,'Side']:
                    case 'R':
                        right = True
                    case 'L':
                        right = False
                    case _:
                        self.s_print('No 9HP side found')

            #NOTE: flips y data if left hand....idk if i want to keep this
            if not right:
                data['THU Y'] = data['THU Y']*-1
                data['POI Y'] = data['POI Y']*-1

            d_dt_data = pd.DataFrame(np.gradient(data,1/self.marker_fs,axis=0),columns=data.columns)
            threshold = 1
            v_peaks = signal.find_peaks(d_dt_data['THU Y'],height = threshold, distance=50)
            v_troughs = signal.find_peaks(-1*d_dt_data['THU Y'],height = threshold, distance=50)

            # for right side, find the first positive peak
            t_start_peak = []
            t_stop_peak = []
            for peakloc,amp in zip(v_peaks[0],v_peaks[1]['peak_heights']):
                thresh = amp * 0.05

                #left threshold
                for i in range(peakloc,0,-1):
                    if np.sum(d_dt_data['THU Y'][i:(i+5)] < thresh) == 1:
                        t_start_peak.append(i)
                        break
                    
                #right threshold 
                for i in range(peakloc,data.shape[0],1):
                    if np.sum(d_dt_data['THU Y'][i:(i+5)] < thresh) == 1:
                        t_stop_peak.append(i)
                        break

            t_start_trough = []
            t_stop_trough = []
            for peakloc,amp in zip(v_troughs[0],v_troughs[1]['peak_heights']):
                thresh = amp * -0.05

                #left threshold
                for i in range(peakloc,0,-1):
                    if np.sum(d_dt_data['THU Y'][i:(i+5)] > thresh) == 1:
                        t_start_trough.append(i)
                        break
                    
                #right threshold 
                for i in range(peakloc,data.shape[0],1):
                    if np.sum(d_dt_data['THU Y'][i:(i+5)] > thresh) == 1:
                        t_stop_trough.append(i)
                        break

            # dont include the last one because it doesnt have a corresponding return
            l=np.min([len(t_start_trough),len(t_start_peak)])                
            t_start_trough = t_start_trough[0:l]
            t_stop_trough = t_stop_trough[0:l]
            t_start_peak = t_start_peak[0:l]
            t_stop_peak = t_stop_peak[0:l]

            # if plot:
            #     plt.subplot(2,1,1)
            #     plt.title('Medial Position')
            #     plt.plot(data['THU Y'])
            #     plt.plot(data['POI Y'])
            #     plt.subplot(2,1,2)
            #     plt.title('Medial Velocity')
            #     plt.plot(d_dt_data['THU Y'])
            #     plt.plot(d_dt_data['POI Y'])
            #     plt.axhline(threshold,color='g',linestyle='--')
            #     plt.axhline(-1*threshold,color='g',linestyle='--')

            #     # plt.axvline(t_start_peak,color = 'pink',linestyle = '--')
            #     plt.vlines(t_start_peak,-1*np.min(v_troughs[1]['peak_heights']),np.max(v_peaks[1]['peak_heights']),color='black',linestyle='--',alpha=0.5)
            #     plt.vlines(t_stop_peak,-1*np.min(v_troughs[1]['peak_heights']),np.max(v_peaks[1]['peak_heights']),color='black',linestyle='--',alpha=0.5)

            #     plt.vlines(t_start_trough,-1*np.min(v_troughs[1]['peak_heights']),np.max(v_troughs[1]['peak_heights']),color='black',linestyle='--',alpha=0.5)
            #     plt.vlines(t_stop_trough,-1*np.min(v_troughs[1]['peak_heights']),np.max(v_troughs[1]['peak_heights']),color='black',linestyle='--',alpha=0.5)
            #     # plt.axvline(t_stop,color = 'pink',linestyle = '--')
            #     plt.legend(['Thumb','Pointer'])
            #     plt.scatter(v_peaks[0],v_peaks[1]['peak_heights'],color='r')
            #     plt.scatter(v_troughs[0],-1*v_troughs[1]['peak_heights'],color='r')
            #     plt.show()

            # ANALYSIS
            # Total Movement Time
            total_movement_time = (t_stop_peak[-1] - t_start_peak[0]) / self.marker_fs
            if right:
                self.output_data.addData("9HPR","TMT",np.array(total_movement_time))
            else:
                self.output_data.addData("9HPL","TMT",np.array(total_movement_time))

            # Single Peg Movement Time
            single_peg_movement_time = []
            for i in range(len(t_start_peak)-1):
                single_peg_movement_time.append((t_start_peak[i+1]-t_start_peak[i])/self.marker_fs)
            if right:
                self.output_data.addData("9HPR","Single Peg MT",np.array(single_peg_movement_time))
            else:
                self.output_data.addData("9HPL","Single Peg MT",np.array(single_peg_movement_time))

            # Gasp-Reach Ratio
            #TODO: need to find the time when they actually are done putting the peg in the hole...could do this with the x data?
            #TODO: check if this is right
            gr_ratio = []
            for i in range(len(t_start_peak)-1):
                gr = ((t_start_trough[i]-t_stop_peak[i]) + (t_start_peak[i+1] - t_stop_trough[i])) / ((t_stop_peak[i] - t_start_peak[i]) + (t_stop_trough[i]-t_start_trough[i])) 
                gr_ratio.append(gr)
            if right:
                self.output_data.addData("9HPR","GR Ratio",np.array(gr_ratio))
            else:
                self.output_data.addData("9HPL","GR Ratio",np.array(gr_ratio))           
            # print(gr_ratio)

            # Peak Speed
            peak_speeds = v_peaks[1]['peak_heights']
            if right:
                self.output_data.addData("9HPR","Peak Speeds",peak_speeds)
            else:
                self.output_data.addData("9HPL","Peak Speeds",peak_speeds)           
            
            # Trough Speed
            #TODO: should i assign this to something different?
            trough_speeds=v_troughs[1]['peak_heights']
            if right:
                self.output_data.addData("9HPR","Peak Speeds",trough_speeds)
            else:
                self.output_data.addData("9HPL","Peak Speeds",trough_speeds)
                
            # Time to peak speed
            time_to_peak_speeds = []
            for tinit,tpeak in zip(t_start_peak,v_peaks[0]):
                time_to_peak_speeds.append((tpeak - tinit)/self.marker_fs)
            for tinit,tpeak in zip(t_start_trough,v_troughs[0]):
                time_to_peak_speeds.append((tpeak - tinit)/self.marker_fs)
            if right:
                self.output_data.addData("9HPR","Peak Speed Time",np.array(time_to_peak_speeds))
            else:
                self.output_data.addData("9HPL","Peak Speed Time",np.array(time_to_peak_speeds))

            # Tangential Velocity Smoothness
            #TODO: need to figure out which is return and transfer
            pointer_v = np.abs(d_dt_data['POI X']) + np.abs(d_dt_data['POI Y']) + np.abs(d_dt_data['POI Z'])
            thumb_v = np.abs(d_dt_data['THU X']) + np.abs(d_dt_data['THU Y']) + np.abs(d_dt_data['THU Z'])

            # Jerk
            for sstart,sstop in zip(t_start_peak,t_stop_peak):
                nj,_ = normJerk(data['POI X'][sstart:sstop],data['POI Y'][sstart:sstop],data['POI Z'][sstart:sstop],self.marker_fs)
                if right:
                    self.output_data.addData("9HPR","Peak Jerk",nj)
                else:
                    self.output_data.addData("9HPL","Peak Jerk",nj)

            if plot:
                if right:
                    axx = axs_right
                else:
                    axx = axs_left
                for i in range(len(t_start_peak)):
                    if i >= len(t_stop_trough) or i >= len(t_start_peak) or i >= len(t_start_trough) or i >= len(t_stop_peak):
                        continue
                    # plt.subplot(2,2,1)
                    t = np.linspace(0,1,t_stop_peak[i]-t_start_peak[i])
                    axx[0,0].plot(t,pointer_v[t_start_peak[i]:t_stop_peak[i]])
                    axx[0,0].set_title("Transfer")
                    axx[0,0].set_ylabel("Pointer")
                    # axs2.subplot(2,2,3)
                    axx[1,0].plot(t,thumb_v[t_start_peak[i]:t_stop_peak[i]])
                    axx[1,0].set_ylabel("Thumb")
                    # axs2.subplot(2,2,2)
                    t = np.linspace(0,1,t_stop_trough[i]-t_start_trough[i])
                    axx[0,1].plot(t,pointer_v[t_start_trough[i]:t_stop_trough[i]])
                    axx[0,1].set_title("Return")
                    # axs2.subplot(2,2,4)
                    axx[1,1].plot(t,thumb_v[t_start_trough[i]:t_stop_trough[i]])
                # fig2.ylabel('Thumb')
                # fig2.ylabel('Pointer')
                if right:
                    fig_right.suptitle(self.id+"\nRight")
                else:
                    fig_left.suptitle(self.id+"\nLeft")
        if plot:
            plt.show()

    def analyze_NG(self,plot):
        if self.id == '002':
            return
        ng_keys = getIndexNames('NG',self.marker_task_labels)
        
        list_of_ends = []
        pplot_avg = np.zeros(200)
        tplot_avg = np.zeros(200)
        for i in range(len(ng_keys)):
            data = self.marker_data[ng_keys[i]]

            # This is a really stupid way to fix the labelling
            if data.columns[9][0] != 'R' and data.columns[9][0] != 'L':
                data['RPOI X'] = data['POI X']
                data['RPOI Y'] = data['POI Y']
                data['RPOI Z'] = data['POI Z']
                data['RTHU X'] = data['THU X']
                data['RTHU Y'] = data['THU Y']
                data['RTHU Z'] = data['THU Z']

            elif self.marker_task_labels['Side'][ng_keys[i]] == 'L':
                data['RPOI X'] = data['LPOI X']
                data['RPOI Y'] = data['LPOI Y']
                data['RPOI Z'] = data['LPOI Z']
                data['RTHU X'] = data['LTHU X']
                data['RTHU Y'] = data['LTHU Y']
                data['RTHU Z'] = data['LTHU Z']

            if self.id == '003':
                list_of_starts = [98,90,88,80,98,46,70,83,70,57]
                start = list_of_starts[i]
                list_of_ends = [228,204,194,180,197,170,189,205,179,164]
                end = list_of_ends[i]
                cc = '#d8b365'
            elif self.id == '001':
                list_of_starts = [57,41,32,35,31,29,26,35,37,34]
                start = list_of_starts[i]
                list_of_ends = [155,141,122,138,135,137,118,138,132,137]
                end = list_of_ends[i]
                cc = '#5ab4ac'
            elif self.id == '004':
                list_of_starts = [64,128,126,55,66,52,60,51,48,66]
                start = list_of_starts[i]
                list_of_ends = [174,245,245,171,182,156,162,171,165,191]
                end = list_of_ends[i]
                cc = '#d8b365'
            elif self.id =='005':
                list_of_starts = [152,196,183,142,229,175,168]
                start = list_of_starts[i]
                list_of_ends = [276,303,296,270,380,303,295]
                end = list_of_ends[i]
                cc = '#d8b365'
            elif self.id =='006':
                list_of_starts = [90,115,100,120,125,100,100,115,95,95]   #100 -> 171 could be shit
                start = list_of_starts[i]
                list_of_ends = [177,227,200,210,210,171,188,191,178,171]
                end = list_of_ends[i]
                cc = '#d8b365'

            else:
                plt.plot(data['RPOI Z'])
                plt.show()
                start = 0
                end = int(input("where to truncate: "))
                list_of_ends.append(end)
            pstart = ( data['RPOI X'][0],data['RPOI Y'][0],data['RPOI Z'][0])
            tstart = ( data['RTHU X'][0],data['RTHU Y'][0],data['RTHU Z'][0])

            p_old = pstart
            t_old = tstart
            p_dist = 0
            t_dist = 0
            mga = 0
            mga_i = 0
            for j in range(end):
                pxyz = ( data['RPOI X'][j], data['RPOI Y'][j], data['RPOI Z'][j])
                txyz = ( data['RTHU X'][j], data['RTHU Y'][j], data['RTHU Z'][j])
                
                p_dist += distance3D(p_old,pxyz)    
                t_dist += distance3D(t_old,txyz)    
                p_old = pxyz
                t_old = txyz
                #mga
                ga = distance3D(pxyz,txyz)
                if ga > mga:
                    mga = ga
                    mga_i = j
            
            #calculate accel/jerk
            p_accel,p_jerk,pplot_stuff = pathAccelJerk(data['RPOI X'][start:end],data['RPOI Y'][start:end],data['RPOI Z'][start:end],self.marker_fs)
            t_accel,t_jerk,tplot_stuff = pathAccelJerk(data['RTHU X'][start:end],data['RTHU Y'][start:end],data['RTHU Z'][start:end],self.marker_fs)                      
            pplot_avg = np.vstack((pplot_avg,pplot_stuff))
            tplot_avg = np.vstack((tplot_avg,tplot_stuff))
            if plot:
                plt.subplot(1,2,1)
                plt.plot(np.linspace(0,1,pplot_stuff.size),pplot_stuff,color = 'grey',alpha=0.4)
                plt.subplot(1,2,2)
                plt.plot(np.linspace(0,1,pplot_stuff.size),tplot_stuff,color = 'grey',alpha=0.4)
            
            # #calculate path length
            p_ideal_distance = distance3D(pstart,pxyz)
            t_ideal_distance = distance3D(tstart,txyz)
            # self.marker_output_data.loc[ng_keys[i],'Pointer Dist'] = p_dist / p_ideal_distance
            # self.marker_output_data.loc[ng_keys[i],'Thumb Dist'] = t_dist / t_ideal_distance

            #new save path length
            self.output_data.addData('NG','Pointer Dist',np.array(p_dist/p_ideal_distance))
            self.output_data.addData('NG','Thumb Dist',np.array(t_dist/t_ideal_distance))

            # # save accel, jerk
            # self.marker_output_data.loc[ng_keys[i],'Pointer Accel'] = p_accel
            # self.marker_output_data.loc[ng_keys[i],'Thumb Accel'] = t_accel
            # self.marker_output_data.loc[ng_keys[i],'Pointer Jerk'] = p_jerk
            # self.marker_output_data.loc[ng_keys[i],'Thumb Jerk'] = t_jerk

            # new save accel, jerk
            self.output_data.addData('NG','Pointer Accel',np.array(p_accel))
            self.output_data.addData('NG','Thumb Accel',np.array(t_accel))
            self.output_data.addData('NG','Pointer Jerk',np.array(p_jerk))
            self.output_data.addData('NG','Thumb Jerk',np.array(t_jerk))

            # # save MGA
            # self.marker_output_data.loc[ng_keys[i],'MGA'] = mga
            # self.marker_output_data.loc[ng_keys[i],'MGA_t'] = mga_i/end

            # new save MGA
            self.output_data.addData('NG','MGA',np.array(mga))
            self.output_data.addData('NG','MGA_t',np.array(mga_i/end))

        pplot_avg = np.delete(pplot_avg, (0), axis=0)
        tplot_avg = np.delete(tplot_avg, (0), axis=0)

        pplot_std = np.std(pplot_avg,axis=0)
        tplot_std = np.std(tplot_avg,axis=0)
        pplot_mean = np.mean(pplot_avg,axis=0)
        tplot_mean = np.mean(tplot_avg,axis=0)
        
        if plot:
            plt.subplot(1,2,1)
            plt.plot(np.linspace(0,1,pplot_stuff.size),pplot_mean,linewidth=2,color= 'black')
            plt.fill_between(np.linspace(0,1,pplot_stuff.size),pplot_mean-pplot_std,pplot_mean+pplot_std,color=cc)
            plt.xlabel('Pointer Fraction of Movement Completion')
            plt.ylabel('Movement Speed\n$mm/s$')
            plt.subplot(1,2,2)
            plt.plot(np.linspace(0,1,pplot_stuff.size),tplot_mean,linewidth=2,color='black')
            plt.fill_between(np.linspace(0,1,tplot_stuff.size),tplot_mean-tplot_std,tplot_mean+tplot_std,color=cc)
            plt.xlabel('Thumb Fraction of Movement Completion')
            plt.show()
            ax = plt.axes(projection ='3d')
            # ax.scatter(block_x,block_y,block_z)

        for i in range(len(ng_keys)):
            data = self.marker_data[ng_keys[i]]
            end = list_of_ends[i]
            if plot:
                if self.marker_task_labels['Side'][ng_keys[i]] == 'R':
                    ax.plot3D(data['RPOI X'][:end],data['RPOI Y'][:end],data['RPOI Z'][:end],c='purple',alpha=0.7)
                    ax.plot3D(data['RTHU X'][:end],data['RTHU Y'][:end],data['RTHU Z'][:end],c='g',alpha=0.7)
                else:
                    if 'LPOI X' in data.columns:
                        ax.plot3D(data['LPOI X'][:end],data['LPOI Y'][:end],data['LPOI Z'][:end],c='b',alpha=0.7)
                        ax.plot3D(data['LTHU X'][:end],data['LTHU Y'][:end],data['LTHU Z'][:end],c='r',alpha=0.7)
                    else:
                        ax.plot3D(data['POI X'][:end],data['POI Y'][:end],data['POI Z'][:end],c='b',alpha=0.7)
                        ax.plot3D(data['THU X'][:end],data['THU Y'][:end],data['THU Z'][:end],c='r',alpha=0.7)

        if plot:
            plt.show()

        for i in range(len(ng_keys)):
                    data = self.marker_data[ng_keys[i]]
                    end = list_of_ends[i]
                    if plot:
                        if self.marker_task_labels['Side'][ng_keys[i]] == 'R':
                            plt.plot(data['RPOI X'][:end],data['RPOI Y'][:end],c='purple',alpha=0.7)
                            plt.plot(data['RTHU X'][:end],data['RTHU Y'][:end],c='g',alpha=0.7)
                        else:
                            if 'LPOI X' in data.columns:
                                plt.plot(data['LPOI X'][:end],data['LPOI Y'][:end],c='b',alpha=0.7)
                                plt.plot(data['LTHU X'][:end],data['LTHU Y'][:end],c='r',alpha=0.7)
                            else:
                                plt.plot(data['POI X'][:end],data['POI Y'][:end],c='b',alpha=0.7)
                                plt.plot(data['THU X'][:end],data['THU Y'][:end],c='r',alpha=0.7)
        if plot:
            plt.show()
         

    def calculate_joint_angles_walking(self,plot: bool):
        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        lh = np.empty(0)
        rh = np.empty(0)
        lk = np.empty(0)
        rk = np.empty(0)
        for i in range(len(walking_keys)):
            data = getMarkerlessData(self.markerless_data,walking_keys[i],['Right Knee AnglesX','Left Knee AnglesX','Right Hip AnglesX','Left Hip AnglesX'])
            t=np.linspace(0,data.shape[0]/self.markerless_fs,num=data.shape[0])
            #right knee
            r_knee = data['Right Knee AnglesX'].values
            r_knee_peaks = signal.find_peaks(r_knee,height=35)

            #left knee
            l_knee = data['Left Knee AnglesX'].values
            l_knee_peaks = signal.find_peaks(l_knee,height=35)

            hip_height = 10
            hip_distance = 50
            #right hip
            r_hip = data['Right Hip AnglesX'].values
            r_hip_peaks = signal.find_peaks(r_hip, height=hip_height,distance = hip_distance)

            #left hip
            l_hip=data['Left Hip AnglesX'].values
            l_hip_peaks = signal.find_peaks(l_hip,height=hip_height,distance=hip_distance)

            rk=np.append(rk,r_knee[r_knee_peaks[0]])
            lk=np.append(lk,l_knee[l_knee_peaks[0]])
            rh=np.append(rh,r_hip[r_hip_peaks[0]])
            lh=np.append(lh,l_hip[l_hip_peaks[0]])

            self.output_data.addData('Walking','Knee_Angle_R',r_knee[r_knee_peaks[0]])
            self.output_data.addData('Walking','Knee_Angle_L',l_knee[l_knee_peaks[0]])
            self.output_data.addData('Walking','Hip_Angle_R',r_hip[r_hip_peaks[0]])
            self.output_data.addData('Walking','Hip_Angle_L',l_hip[l_hip_peaks[0]])
            # print(walking_keys[i])
            # print(l_hip[l_hip_peaks[0]]) 
            if plot:
                plt.subplot(2,2,1)
                plt.plot(l_hip)
                plt.scatter(l_hip_peaks[0],l_hip[l_hip_peaks[0]],c='r')
                plt.title('Left Hip Angle')
                plt.ylabel('Angle (degrees)')
                
                plt.subplot(2,2,2)
                plt.plot(l_knee)
                plt.scatter(l_knee_peaks[0],l_knee[l_knee_peaks[0]],c='r')
                plt.title('Left Knee Angle')

                plt.subplot(2,2,3)
                plt.plot(r_hip)
                plt.scatter(r_hip_peaks[0],r_hip[r_hip_peaks[0]],c='r')
                plt.title('Right Hip Angle')
                plt.ylabel('Angle (degrees)')
                plt.xlabel('Time (samples)')
                
                plt.subplot(2,2,4)
                plt.plot(r_knee)
                plt.scatter(r_knee_peaks[0],r_knee[r_knee_peaks[0]],c='r')
                plt.title('Right Knee Angle')
                plt.xlabel('Time (samples)')

                plt.suptitle(walking_keys[i])
                plt.show()

        if plot:
            print("Means:\nRH:\t%.2f\nLH:\t%.2f\nRK:\t%.2f\nLK:\t%.2f\n"%(rh.mean(),lh.mean(),rk.mean(),lk.mean()))


    def calculate_step_width(self,plot: bool):
        """Uses the RTO,LTO,etc from V3D"""
        # plot = True
        peakdict2 = {}
        step_width = []
        step_length = []
        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        for trial in walking_keys:
            data = self.markerless_data[trial]
            rhs = np.array(self.markerless_events[trial]['RHS'])
            lhs = np.array(self.markerless_events[trial]['LHS'])
            rto = np.array(self.markerless_events[trial]['RTO'])
            lto = np.array(self.markerless_events[trial]['LTO'])
            rhs = np.round(rhs*self.markerless_fs).astype(int)
            lhs = np.round(lhs*self.markerless_fs).astype(int)
            rto = np.round(rto*self.markerless_fs).astype(int)
            lto = np.round(lto*self.markerless_fs).astype(int)
            max_ind = np.max(data.index)
            rhs = rhs[np.where(rhs<=max_ind)]
            lhs = lhs[np.where(lhs<=max_ind)]
            rto = rto[np.where(rto<=max_ind)]
            lto = lto[np.where(lto<=max_ind)]

            if plot:
                plt.subplot(3,1,2)
                plt.plot(data['R_HEELX'],data['R_HEELY'],'b',data['L_HEELX'],data['L_HEELY'],'g')
                plt.scatter(data['R_HEELX'][rhs],data['R_HEELY'][rhs],c='r')
                plt.scatter(data['L_HEELX'][lhs],data['L_HEELY'][lhs],c='y')
                plt.scatter(data['R_HEELX'][rto],data['R_HEELY'][rto],c='magenta')
                plt.scatter(data['L_HEELX'][lto],data['L_HEELY'][lto],c='orange')
                
                # plt.plot(data['R_HEELX'],data['R_HEELY'],'b',data['L_HEELX'],data['L_HEELY'],'g')
                # plt.scatter(rhs,data['R_HEELY'][rhs],c='r')
                # plt.scatter(lhs,data['L_HEELY'][lhs],c='y')
                # plt.scatter(rto,data['R_HEELY'][rto],c='magenta')
                # plt.scatter(lto,data['L_HEELY'][lto],c='orange')
                plt.legend(['Right Heel','Left Heel','RHS','LHS','RTO','LTO'])
                plt.ylabel('Lab Y\n(Across)')
                xminlim,xmaxlim = plt.xlim()

                plt.subplot(3,1,1)
                plt.plot(data['L_HEELX'],data['L_HEELZ'],'g')
                plt.scatter(data['L_HEELX'][lhs],data['L_HEELZ'][lhs],c='y')
                plt.scatter(data['L_HEELX'][rhs],data['L_HEELZ'][rhs],c='r')
                plt.scatter(data['L_HEELX'][rto],data['L_HEELZ'][rto],c='magenta')
                plt.scatter(data['L_HEELX'][lto],data['L_HEELZ'][lto],c='orange')
                # plt.plot(data['L_HEELZ'],'g')
                # plt.scatter(lhs,data['L_HEELZ'][lhs],c='y')

                plt.scatter(rhs,data['L_HEELZ'][rhs],c='r')
                plt.scatter(lhs,data['L_HEELZ'][lhs],c='y')
                plt.scatter(rto,data['L_HEELZ'][rto],c='magenta')
                plt.scatter(lto,data['L_HEELZ'][lto],c='orange')

                plt.ylabel('Lab Z\n(Vertical)')
                plt.xlim([xminlim,xmaxlim])

                plt.subplot(3,1,3)
                plt.plot(data['R_HEELX'],data['R_HEELZ'],'b')
                plt.scatter(data['R_HEELX'][rhs],data['R_HEELZ'][rhs],c='r')
                # plt.plot(data['R_HEELZ'],'b')
                # plt.scatter(rhs,data['R_HEELZ'][rhs],c='r')

                plt.scatter(data['R_HEELX'][lhs],data['R_HEELZ'][lhs],c='y')
                plt.scatter(data['R_HEELX'][rto],data['R_HEELZ'][rto],c='magenta')
                plt.scatter(data['R_HEELX'][lto],data['R_HEELZ'][lto],c='orange')

                plt.ylabel('Lab Z\n(Vertical)')
                plt.xlabel('Lab X\n(Along)')
                plt.xlim([xminlim,xmaxlim])
                print(trial)
                print(rhs)
                print(lhs)
                print(rto)
                print(lto)
                plt.suptitle(trial + '\n' + self.id+'\nV3D')

            r_peaks = rhs
            l_peaks = lhs
            for _ in range(r_peaks.size+l_peaks.size):
                if r_peaks.size ==0 or l_peaks.size ==0:
                    break
                r_min = np.min(r_peaks)
                l_min = np.min(l_peaks)
                if plot:
                    plt.subplot(3,1,2)
                    plt.plot([data['R_HEELX'].values[r_peaks[0]],data['L_HEELX'].values[l_peaks[0]]], [data['R_HEELY'].values[r_peaks[0]],data['L_HEELY'].values[l_peaks[0]]], 'pink', linestyle="--")
                step_length.append(np.abs(data['R_HEELX'].values[r_peaks[0]]-data['L_HEELX'].values[l_peaks[0]]))
                step_width.append(np.abs(data['R_HEELY'].values[r_peaks[0]]-data['L_HEELY'].values[l_peaks[0]]))
                if r_min < l_min:
                    r_peaks = np.delete(r_peaks,0)
                else:
                    l_peaks = np.delete(l_peaks,0)
            if plot:
                print("\nStep Width: %.3f +/- (%.3f)"%(np.mean(step_width),np.std(step_width)))
                print("Step length: %.3f +/- (%.3f)"%(np.mean(step_length),np.std(step_length)))
                plt.show()
            self.output_data.addData('Walking','step_length',np.array(step_length)/self.height)
            self.output_data.addData('Walking','step_width',np.array(step_width)/self.height)
            step_length = []
            step_width = []
            peakdict2.update({trial:[rhs,lhs,rto,lto]})

        self.peakdict2 = peakdict2

    def calculate_step_height(self,plot:bool):
        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        step_height = []
        for i in range(len(walking_keys)):
            data = getMarkerlessData(self.markerless_data,walking_keys[i],['L_HEELZ','R_HEELZ'])
            # L Heel
            l_peaks = signal.find_peaks(data['L_HEELZ'].values,height=0.1)[0]
            if plot:
                plt.subplot(1,2,1)
                plt.plot(data['L_HEELZ'].values)
                plt.scatter(l_peaks,data['L_HEELZ'].values[l_peaks],c='r')
                plt.ylabel('Heel Vertical Position (m)')
                plt.xlabel('Time (samples)')

            #R Heel
            r_peaks = signal.find_peaks(data['R_HEELZ'].values,height=0.1)[0]
            if plot:
                plt.subplot(1,2,2)
                plt.plot(data['R_HEELZ'].values)
                plt.scatter(r_peaks,data['R_HEELZ'].values[r_peaks],c='r')
                plt.suptitle(walking_keys[i])
            
            for j in l_peaks:
                step_height.append(data['L_HEELZ'].values[j])
            for j in r_peaks:
                step_height.append(data['R_HEELZ'].values[j])
            if plot:
                print("Step height: %.3f +/- (%.3f)"%(np.mean(step_height),np.std(step_height)))
                plt.show()
            self.output_data.addData('Walking','step_height',np.array(step_height)/self.height)
            step_height = []

    def calculate_pelvis_jerk_step(self,plot:bool):
        self.markerless_output_data['Pelvis_Jerk'] = np.nan

        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        for i in range(len(walking_keys)):
            rhs = self.peakdict2[walking_keys[i]][0]
            lhs = self.peakdict2[walking_keys[i]][1]
            data = getMarkerlessData(self.markerless_data,walking_keys[i],['PelvisPosX','PelvisPosY','PelvisPosZ'])

            if lhs.min() < rhs.min():
                side_heel = lhs
                anti_heel = rhs
            else:
                side_heel = rhs
                anti_heel = lhs
            
            jerks = []
            for _ in range(side_heel.size):
                if side_heel.size == 0 or anti_heel.size==0: 
                    break
                if anti_heel[0] < side_heel[0]:
                    self.s_print("["+walking_keys[i]+"] Possible Error")

                minind = side_heel[0]
                maxind = anti_heel[0]
                if minind < data.index[0]:
                    continue
                
                nj, plotstuff = normJerk(data.loc[minind:maxind,'PelvisPosX'],data.loc[minind:maxind,'PelvisPosY'],data.loc[minind:maxind,'PelvisPosZ'],self.markerless_fs)

                if plot:
                    t = np.linspace(0,1,num=maxind-minind+1)
                    plt.subplot(4,3,1)
                    plt.plot(t,data.loc[minind:maxind,'PelvisPosX'].values)
                    plt.subplot(4,3,2)
                    plt.plot(t,data.loc[minind:maxind,'PelvisPosY'].values)
                    plt.subplot(4,3,3)
                    plt.plot(t,data.loc[minind:maxind,'PelvisPosZ'].values)
                    plt.subplot(4,3,4)
                    plt.plot(t,plotstuff[0])
                    plt.subplot(4,3,5)
                    plt.plot(t,plotstuff[3])
                    plt.subplot(4,3,6)
                    plt.plot(t,plotstuff[6])

                    plt.subplot(4,3,7)
                    plt.plot(t,plotstuff[1])
                    plt.subplot(4,3,8)
                    plt.plot(t,plotstuff[4])
                    plt.subplot(4,3,9)
                    plt.plot(t,plotstuff[7])

                    plt.subplot(4,3,10)
                    plt.plot(t,plotstuff[2])
                    plt.subplot(4,3,11)
                    plt.plot(t,plotstuff[5])
                    plt.subplot(4,3,12)
                    plt.plot(t,plotstuff[8])
                    plt.suptitle("Pelvis Kinematics over Step\n"+self.id)

                    # #this is for just the velocity profiles
                    # t = np.linspace(0,1,num=maxind-minind+1)
                    # if self.id == '002':
                    #     plt.plot(t,plotstuff[6],color='#130AF1')
                    # elif self.id == '003':
                    #     plt.plot(t,plotstuff[6],color='#FE8821')


                jerks.append(nj)
                side_heel = np.delete(side_heel,0)
                tmp = anti_heel
                anti_heel = side_heel
                side_heel = tmp
            self.output_data.addData('Walking','pelvis_jerk_step_normalized',np.array(jerks))
        if plot:
            # normal
            plt.subplot(4,3,1)
            plt.ylabel('Position')
            plt.subplot(4,3,4)
            plt.ylabel('Veloctiy')
            plt.subplot(4,3,7)
            plt.ylabel('Acceleration')
            plt.subplot(4,3,10)
            plt.xlabel('Time (fraction)\nX')
            plt.ylabel('Jerk')
            plt.subplot(4,3,11)
            plt.xlabel('Time (fraction)\nY')
            plt.subplot(4,3,12)
            plt.xlabel('Time (fraction)\nZ')
            plt.show()
            plt.hist(jerks)
            plt.show()

            # #just velocity
            # plt.ylim([-0.6,0.6])
            # plt.title('Walking\n'+self.id)
            # plt.show()

    def calculate_thorax_jerk_step(self,plot:bool):
        self.markerless_output_data['Thorax_Jerk'] = np.nan
        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        for i in range(len(walking_keys)):
            rhs = self.peakdict2[walking_keys[i]][0]
            lhs = self.peakdict2[walking_keys[i]][1]
            data = getMarkerlessData(self.markerless_data,walking_keys[i],['Distal ThoraxX','Distal ThoraxY','Distal ThoraxZ'])

            if lhs.min() < rhs.min():
                side_heel = lhs
                anti_heel = rhs
            else:
                side_heel = rhs
                anti_heel = lhs
            
            jerks = []
            self.thorax_z_velocity_profiles=[]
            for _ in range(side_heel.size):
                if side_heel.size == 0 or anti_heel.size==0: 
                    break
                if anti_heel[0] < side_heel[0]:
                    self.s_print("["+walking_keys[i]+"] Possible Error")

                minind = side_heel[0]
                maxind = anti_heel[0]
                if minind < data.index[0]:
                    continue

                nj, plotstuff = normJerk(data.loc[minind:maxind,'Distal ThoraxX'],data.loc[minind:maxind,'Distal ThoraxY'],data.loc[minind:maxind,'Distal ThoraxZ'],self.markerless_fs)

                if plot:
                    t = np.linspace(0,1,num=maxind-minind+1)
                    plt.subplot(4,3,1)
                    plt.plot(t,data.loc[minind:maxind,'Distal ThoraxX'].values)
                    plt.subplot(4,3,2)
                    plt.plot(t,data.loc[minind:maxind,'Distal ThoraxY'].values)
                    plt.subplot(4,3,3)
                    plt.plot(t,data.loc[minind:maxind,'Distal ThoraxZ'].values)
                    plt.subplot(4,3,4)
                    plt.plot(t,plotstuff[0])
                    plt.subplot(4,3,5)
                    plt.plot(t,plotstuff[3])
                    plt.subplot(4,3,6)
                    plt.plot(t,plotstuff[6])
                    self.thorax_z_velocity_profiles.append(plotstuff[6])

                    plt.subplot(4,3,7)
                    plt.plot(t,plotstuff[1])
                    plt.subplot(4,3,8)
                    plt.plot(t,plotstuff[4])
                    plt.subplot(4,3,9)
                    plt.plot(t,plotstuff[7])

                    plt.subplot(4,3,10)
                    plt.plot(t,plotstuff[2])
                    plt.subplot(4,3,11)
                    plt.plot(t,plotstuff[5])
                    plt.subplot(4,3,12)
                    plt.plot(t,plotstuff[8])
                    plt.suptitle("Thorax Kinematics over Step\n"+self.id)
                jerks.append(nj)
                side_heel = np.delete(side_heel,0)
                tmp = anti_heel
                anti_heel = side_heel
                side_heel = tmp
            self.output_data.addData('Walking','thorax_jerk_step_normalized',np.array(jerks))
        if plot:
            plt.subplot(4,3,1)
            plt.ylabel('Position')
            plt.subplot(4,3,4)
            plt.ylabel('Veloctiy')
            plt.subplot(4,3,7)
            plt.ylabel('Acceleration')
            plt.subplot(4,3,10)
            plt.xlabel('Time (fraction)\nX')
            plt.ylabel('Jerk')
            plt.subplot(4,3,11)
            plt.xlabel('Time (fraction)\nY')
            plt.subplot(4,3,12)
            plt.xlabel('Time (fraction)\nZ')
            plt.show()
            plt.hist(jerks)
            plt.show()

    def calculate_support(self,plot:bool):
        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        double_stances = []
        gait_cycle_duration = []
        for trial in walking_keys:
            rhs = self.peakdict2[trial][0]
            lhs = self.peakdict2[trial][1]
            rto = self.peakdict2[trial][2]
            lto = self.peakdict2[trial][3]
            data = getMarkerlessData(self.markerless_data,trial,['R_HEELZ','L_HEELZ'])

            if lhs.min() < rhs.min():
                side_heel = lhs
                anti_toe = rto
                anti_heel = rhs
                side_toe = lto
            else:
                side_heel = rhs
                anti_toe = lto
                anti_heel = lhs
                side_toe = rto
            
            if anti_toe[0] < side_heel[0]:
                anti_toe=np.delete(anti_toe,0)
            if side_toe[0] < side_heel[0]:
                side_toe = np.delete(side_toe,0)
            if plot:
                t=np.linspace(0,data['R_HEELZ'].values.size/self.markerless_fs,data['R_HEELZ'].size)
                offset = data['R_HEELZ'].index[0]
                plt.plot(t,data['R_HEELZ'])
                plt.plot(t,data['L_HEELZ'])
            for _ in range(side_heel.size):

                if side_heel.size == 1 or anti_toe.size == 0 or anti_heel.size==0 or side_toe.size==0:
                    break
               
                cycle_time = side_heel[1] - side_heel[0]
                if anti_toe[0] > side_heel[1] or anti_heel[0] > side_heel[1] or side_toe[0] > side_heel[1]:
                    self.s_print("["+trial+"] Possible Error")
                if plot:   
                    plt.axvspan(t[side_heel[0]-offset],t[anti_toe[0]-offset],color='green',alpha=0.2)
                    plt.axvspan(t[anti_toe[0]-offset],t[anti_heel[0]-offset],color='magenta',alpha=0.2)
                    plt.axvspan(t[anti_heel[0]-offset],t[side_toe[0]-offset],color='green',alpha=0.2)
                    plt.axvspan(t[side_toe[0]-offset],t[side_heel[1]-offset],color='magenta',alpha=0.2)
                    plt.axvline(t[side_heel[0]-offset],c='black',linestyle='--')
                    plt.axvline(t[anti_toe[0]-offset],c='black',linestyle='--')
                    plt.axvline(t[anti_heel[0]-offset],c='black',linestyle='--')
                    plt.axvline(t[side_toe[0]-offset],c='black',linestyle='--')
                    plt.axvline(t[side_heel[1]-offset],c='black',linestyle='--')

                double_stance = (anti_toe[0]-side_heel[0])/cycle_time + (side_toe[0]-anti_heel[0])/cycle_time
                single_stance = (anti_heel[0]-anti_toe[0])/cycle_time + (side_heel[1]-side_toe[0])/cycle_time
                if plot:
                    print("Double stance: %.2f"%double_stance)
                    print("Single stance: %.2f"%single_stance)
                side_heel = np.delete(side_heel,0)
                side_toe = np.delete(side_toe,0)
                anti_heel = np.delete(anti_heel,0)
                anti_toe = np.delete(anti_toe,0)
                double_stances.append(double_stance)
                gait_cycle_duration.append(cycle_time/self.markerless_fs)
            self.output_data.addData('Walking','double_stance',np.array(double_stances))
            self.output_data.addData('Walking','gait_cycle_time',np.array(gait_cycle_duration))
            double_stances = []
            gait_cycle_duration = []

            if plot:
                plt.title(self.id+'\n'+trial)
                plt.ylabel('Z Height\n(m)')
                plt.xlabel('Time (s)')
                plt.legend(['R HEEL','L HEEL','Double Stance','Single Stance'])
                plt.show()

    def walking_angle(self,plot):
        self.markerless_output_data['Walking_Angle_Deviation'] = np.nan
        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        for trial in walking_keys:
            data = getMarkerlessData(self.markerless_data,trial,['PelvisPosX','PelvisPosY'])
            start_position = (data.iloc[0,0],data.iloc[0,1])   #XY coords
            end_position = (data.iloc[-1,0],data.iloc[-1,1])
            ang = np.degrees(np.arctan(np.abs(start_position[1]-end_position[1])/np.abs(start_position[0]-end_position[0])))
            self.output_data.addData('Walking','walking_angle_deviation',np.array(ang))
            if plot:
                plt.plot(data['PelvisPosX'],data['PelvisPosY'],alpha=0.5)
                plt.plot([start_position[0],end_position[0]],[start_position[1],end_position[1]],linestyle='--')
                print("Angle: %f"%ang)
        if plot:
            plt.title('Overhead view of walking path\n'+self.id)
            plt.ylabel('Y lab (m)')
            plt.xlabel('X lab (m)')
            plt.ylim([-0.5,0.5])
            plt.show()

    def calculate_pelvis_jerk_tandem(self,plot:bool):
        # plot = True
        tandem_keys = getIndexNames('Tandem',self.markerless_task_labels)
        for i in range(len(tandem_keys)):
            rhs = np.array(self.markerless_events[tandem_keys[i]]['RHS'])
            lhs = np.array(self.markerless_events[tandem_keys[i]]['LHS'])
            rhs = np.round(rhs*self.markerless_fs).astype(int)
            lhs = np.round(lhs*self.markerless_fs).astype(int)
            data = getMarkerlessData(self.markerless_data,tandem_keys[i],['PelvisPosX','PelvisPosY','PelvisPosZ'])
            max_ind = np.max(data.index)
            rhs = rhs[np.where(rhs<=max_ind)]
            lhs = lhs[np.where(lhs<=max_ind)]

            if lhs.min() < rhs.min():
                side_heel = lhs
                anti_heel = rhs
            else:
                side_heel = rhs
                anti_heel = lhs
            
            jerks = []
            for _ in range(side_heel.size):
                if side_heel.size == 0 or anti_heel.size==0: 
                    break
                if anti_heel[0] < side_heel[0]:
                    self.s_print("["+tandem_keys[i]+"] Possible Error")

                minind = side_heel[0]
                maxind = anti_heel[0]
                if minind < data.index[0]:
                    continue
                
                nj, plotstuff = normJerk(data.loc[minind:maxind,'PelvisPosX'],data.loc[minind:maxind,'PelvisPosY'],data.loc[minind:maxind,'PelvisPosZ'],self.markerless_fs)

                if plot:
                    t = np.linspace(0,1,num=maxind-minind+1)
                    plt.subplot(4,3,1)
                    plt.plot(t,data.loc[minind:maxind,'PelvisPosX'].values)
                    plt.subplot(4,3,2)
                    plt.plot(t,data.loc[minind:maxind,'PelvisPosY'].values)
                    plt.subplot(4,3,3)
                    plt.plot(t,data.loc[minind:maxind,'PelvisPosZ'].values)
                    plt.subplot(4,3,4)
                    plt.plot(t,plotstuff[0])
                    plt.subplot(4,3,5)
                    plt.plot(t,plotstuff[3])
                    plt.subplot(4,3,6)
                    plt.plot(t,plotstuff[6])

                    plt.subplot(4,3,7)
                    plt.plot(t,plotstuff[1])
                    plt.subplot(4,3,8)
                    plt.plot(t,plotstuff[4])
                    plt.subplot(4,3,9)
                    plt.plot(t,plotstuff[7])

                    plt.subplot(4,3,10)
                    plt.plot(t,plotstuff[2])
                    plt.subplot(4,3,11)
                    plt.plot(t,plotstuff[5])
                    plt.subplot(4,3,12)
                    plt.plot(t,plotstuff[8])
                    plt.suptitle("Pelvis Kinematics over Tandem Step\n"+self.id)

                    # #this is for just the velocity profiles
                    # t = np.linspace(0,1,num=maxind-minind+1)
                    # if self.id == '002':
                    #     plt.plot(t,plotstuff[6],color='#130AF1')
                    # elif self.id == '003':
                    #     plt.plot(t,plotstuff[6],color='#FE8821')


                jerks.append(nj)
                side_heel = np.delete(side_heel,0)
                tmp = anti_heel
                anti_heel = side_heel
                side_heel = tmp
            self.output_data.addData('Tandem','pelvis_jerk_step_normalized',np.array(jerks))
        if plot:
            # normal
            plt.subplot(4,3,1)
            plt.ylabel('Position')
            plt.subplot(4,3,4)
            plt.ylabel('Veloctiy')
            plt.subplot(4,3,7)
            plt.ylabel('Acceleration')
            plt.subplot(4,3,10)
            plt.xlabel('Time (fraction)\nX')
            plt.ylabel('Jerk')
            plt.subplot(4,3,11)
            plt.xlabel('Time (fraction)\nY')
            plt.subplot(4,3,12)
            plt.xlabel('Time (fraction)\nZ')
            plt.show()
            plt.hist(jerks)
            plt.show()

            # #just velocity
            # plt.title('Tandem\n'+self.id)
            # plt.ylim([-0.2,0.2])
            # plt.show()

    def calculate_thorax_jerk_tandem(self,plot:bool):
        tandem_keys = getIndexNames('Tandem',self.markerless_task_labels)
        for i in range(len(tandem_keys)):
            rhs = np.array(self.markerless_events[tandem_keys[i]]['RHS'])
            lhs = np.array(self.markerless_events[tandem_keys[i]]['LHS'])
            rhs = np.round(rhs*self.markerless_fs).astype(int)
            lhs = np.round(lhs*self.markerless_fs).astype(int)
            data = getMarkerlessData(self.markerless_data,tandem_keys[i],['Distal ThoraxX','Distal ThoraxY','Distal ThoraxZ'])
            max_ind = np.max(data.index)
            rhs = rhs[np.where(rhs<=max_ind)]
            lhs = lhs[np.where(lhs<=max_ind)]

            if lhs.min() < rhs.min():
                side_heel = lhs
                anti_heel = rhs
            else:
                side_heel = rhs
                anti_heel = lhs
            
            jerks = []
            for _ in range(side_heel.size):
                if side_heel.size == 0 or anti_heel.size==0: 
                    break
                if anti_heel[0] < side_heel[0]:
                    self.s_print("["+tandem_keys[i]+"] Possible Error")

                minind = side_heel[0]
                maxind = anti_heel[0]
                if minind < data.index[0]:
                    continue
                
                nj, plotstuff = normJerk(data.loc[minind:maxind,'Distal ThoraxX'],data.loc[minind:maxind,'Distal ThoraxY'],data.loc[minind:maxind,'Distal ThoraxZ'],self.markerless_fs)

                if plot:
                    t = np.linspace(0,1,num=maxind-minind+1)
                    plt.subplot(4,3,1)
                    plt.plot(t,data.loc[minind:maxind,'Distal ThoraxX'].values)
                    plt.subplot(4,3,2)
                    plt.plot(t,data.loc[minind:maxind,'Distal ThoraxY'].values)
                    plt.subplot(4,3,3)
                    plt.plot(t,data.loc[minind:maxind,'Distal ThoraxZ'].values)
                    plt.subplot(4,3,4)
                    plt.plot(t,plotstuff[0])
                    plt.subplot(4,3,5)
                    plt.plot(t,plotstuff[3])
                    plt.subplot(4,3,6)
                    plt.plot(t,plotstuff[6])

                    plt.subplot(4,3,7)
                    plt.plot(t,plotstuff[1])
                    plt.subplot(4,3,8)
                    plt.plot(t,plotstuff[4])
                    plt.subplot(4,3,9)
                    plt.plot(t,plotstuff[7])

                    plt.subplot(4,3,10)
                    plt.plot(t,plotstuff[2])
                    plt.subplot(4,3,11)
                    plt.plot(t,plotstuff[5])
                    plt.subplot(4,3,12)
                    plt.plot(t,plotstuff[8])
                    plt.suptitle("Thorax Kinematics over Tandem Step\n"+self.id)
                jerks.append(nj)
                side_heel = np.delete(side_heel,0)
                tmp = anti_heel
                anti_heel = side_heel
                side_heel = tmp
            self.output_data.addData('Tandem','thorax_jerk_step_normalized',np.array(jerks))
        if plot:
            plt.subplot(4,3,1)
            plt.ylabel('Position')
            plt.subplot(4,3,4)
            plt.ylabel('Veloctiy')
            plt.subplot(4,3,7)
            plt.ylabel('Acceleration')
            plt.subplot(4,3,10)
            plt.xlabel('Time (fraction)\nX')
            plt.ylabel('Jerk')
            plt.subplot(4,3,11)
            plt.xlabel('Time (fraction)\nY')
            plt.subplot(4,3,12)
            plt.xlabel('Time (fraction)\nZ')
            plt.show()
            plt.hist(jerks)
            plt.show()

    def calculate_support_tandem(self,plot:bool):
        tandem_keys = getIndexNames('Tandem',self.markerless_task_labels)
        double_stances = []
        gait_cycle_duration = []
        for trial in tandem_keys:
            data = getMarkerlessData(self.markerless_data,trial,['Distal ThoraxX','Distal ThoraxY','Distal ThoraxZ'])
            rhs = np.array(self.markerless_events[trial]['RHS'])
            lhs = np.array(self.markerless_events[trial]['LHS'])
            rto = np.array(self.markerless_events[trial]['RTO'])
            lto = np.array(self.markerless_events[trial]['LTO'])
            rhs = np.round(rhs*self.markerless_fs).astype(int)
            lhs = np.round(lhs*self.markerless_fs).astype(int)
            rto = np.round(rto*self.markerless_fs).astype(int)
            lto = np.round(lto*self.markerless_fs).astype(int)
            max_ind = np.max(data.index)
            rhs = rhs[np.where(rhs<=max_ind)]
            lhs = lhs[np.where(lhs<=max_ind)]
            rto = rto[np.where(rto<=max_ind)]
            lto = lto[np.where(lto<=max_ind)]


            data = getMarkerlessData(self.markerless_data,trial,['R_HEELZ','L_HEELZ'])

            if lhs.min() < rhs.min():
                side_heel = lhs
                anti_toe = rto
                anti_heel = rhs
                side_toe = lto
            else:
                side_heel = rhs
                anti_toe = lto
                anti_heel = lhs
                side_toe = rto
            
            if anti_toe[0] < side_heel[0]:
                anti_toe=np.delete(anti_toe,0)
            if side_toe[0] < side_heel[0]:
                side_toe = np.delete(side_toe,0)
            if plot:
                t=np.linspace(0,data['R_HEELZ'].values.size/self.markerless_fs,data['R_HEELZ'].size)
                offset = data['R_HEELZ'].index[0]
                plt.plot(t,data['R_HEELZ'])
                plt.plot(t,data['L_HEELZ'])
            for _ in range(side_heel.size):

                if side_heel.size == 1 or anti_toe.size == 0 or anti_heel.size==0 or side_toe.size==0:
                    break
               
                cycle_time = side_heel[1] - side_heel[0]
                if anti_toe[0] > side_heel[1] or anti_heel[0] > side_heel[1] or side_toe[0] > side_heel[1]:
                    self.s_print("["+trial+"] Possible Error")
                if plot:   
                    plt.axvspan(t[side_heel[0]-offset],t[anti_toe[0]-offset],color='green',alpha=0.2)
                    plt.axvspan(t[anti_toe[0]-offset],t[anti_heel[0]-offset],color='magenta',alpha=0.2)
                    plt.axvspan(t[anti_heel[0]-offset],t[side_toe[0]-offset],color='green',alpha=0.2)
                    plt.axvspan(t[side_toe[0]-offset],t[side_heel[1]-offset],color='magenta',alpha=0.2)
                    plt.axvline(t[side_heel[0]-offset],c='black',linestyle='--')
                    plt.axvline(t[anti_toe[0]-offset],c='black',linestyle='--')
                    plt.axvline(t[anti_heel[0]-offset],c='black',linestyle='--')
                    plt.axvline(t[side_toe[0]-offset],c='black',linestyle='--')
                    plt.axvline(t[side_heel[1]-offset],c='black',linestyle='--')

                double_stance = (anti_toe[0]-side_heel[0])/cycle_time + (side_toe[0]-anti_heel[0])/cycle_time
                single_stance = (anti_heel[0]-anti_toe[0])/cycle_time + (side_heel[1]-side_toe[0])/cycle_time
                if plot:
                    print("Double stance: %.2f"%double_stance)
                    print("Single stance: %.2f"%single_stance)
                side_heel = np.delete(side_heel,0)
                side_toe = np.delete(side_toe,0)
                anti_heel = np.delete(anti_heel,0)
                anti_toe = np.delete(anti_toe,0)
                double_stances.append(double_stance)
                gait_cycle_duration.append(cycle_time/self.markerless_fs)
            self.output_data.addData('Tandem','double_stance',np.array(double_stances))
            self.output_data.addData('Tandem','gait_cycle_time',np.array(gait_cycle_duration))
            double_stances = []
            gait_cycle_duration = []

            if plot:
                plt.title('Tandem '+self.id+'\n'+trial)
                plt.ylabel('Z Height\n(m)')
                plt.xlabel('Time (s)')
                plt.legend(['R HEEL','L HEEL','Double Stance','Single Stance'])
                plt.show()
