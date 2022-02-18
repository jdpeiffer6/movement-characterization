import os
import json
from re import L
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mpld3
from scipy import signal
from scipy.interpolate import interp1d
import numpy as np
from functions.helper import readTaskCSV,getIndexNames

class SessionDataObject:
    def __init__(self, path: str,plot: bool,height: float):
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
        self.test: dict[str:pd.DataFrame]

        self.markerless_output_data: pd.DataFrame
        self.marker_output_data: pd.DataFrame

        self.plot = plot
        
        # this path will be the path to each subject's folder and the date format for their session
        # for instance "...\\002\\2022-01-28"
        print("loading data subject %s for date %s\n" % (path[-14:-11], path[-10:]))

        # this uses the fact that all markerless will be 001 and all marker will be 002
        if os.path.isdir(self.path + '_001'):
            self.load_markerless()
        else:
            print("Could not find markerless session for %s"% self.id)
        if os.path.isdir(self.path + '_002'):
            self.load_markers()
        else:
            print("Could not find marker session for %s"% self.id)
        self.analyze_walking()

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
                print(trial_name)
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
            print(name)
        self.marker_task_labels = readTaskCSV(self.path+'_002')
        self.marker_output_data = self.marker_task_labels


    def analyze_walking(self):
        self.calculate_pelvis_pos(self.plot)
        self.calculate_joint_angles_walking(self.plot)
        self.calculate_step_height(True)
        self.calculate_step_width(True)
        # self.calculate_step_width2(True)
        

    def calculate_joint_angles_walking(self,plot: bool):
        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        lh = np.empty(0)
        rh = np.empty(0)
        lk = np.empty(0)
        rk = np.empty(0)
        for i in range(len(walking_keys)):
            data = self.markerless_data[walking_keys[i]]
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

            self.markerless_output_data.loc[walking_keys[i],'Knee_Angle_R'] = r_knee_peaks[0].mean()
            self.markerless_output_data.loc[walking_keys[i],'Knee_Angle_L'] = l_knee_peaks[0].mean()
            self.markerless_output_data.loc[walking_keys[i],'Hip_Angle_R'] = r_hip_peaks[0].mean()
            self.markerless_output_data.loc[walking_keys[i],'Hip_Angle_L'] = l_hip_peaks[0].mean()

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
                # manager = plt.get_current_fig_manager()
                # manager.showMaximized()
                plt.show()
        #TODO: remove this...the data shouldnt be accessed in this way
        self.r_hip_walking = rh   
        self.l_hip_walking = lh
        self.r_knee_walking = rk
        self.l_knee_walking = lk
        if plot:
            print("Means:\nRH:\t%.2f\nLH:\t%.2f\nRK:\t%.2f\nLK:\t%.2f\n"%(rh.mean(),lh.mean(),rk.mean(),lk.mean()))


    def interpolate_walking(self,plot:bool):
        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        peakdict = {}
        for i in range(len(walking_keys)):
            data = self.markerless_data[walking_keys[i]]
            #interpolate that sh(t)
            # right leg
            xmin = min(data['R_HEELX'])
            xmax = max(data['R_HEELX'])
            r_xnew = np.linspace(xmin,xmax,num = data['R_HEELX'].shape[0],endpoint=True)
            x = data['R_HEELX']
            z = data['R_HEELZ']
            f_interp = interp1d(x,z)
            r_znew = f_interp(r_xnew)
            self.markerless_data[walking_keys[i]]['R_HEELZ_WRT_X'] = r_znew
            r_peaks = signal.find_peaks(-1*r_znew,threshold = -0.05,distance=50)

            # left leg
            xmin = min(data['L_HEELX'])
            xmax = max(data['L_HEELX'])
            l_xnew = np.linspace(xmin,xmax,num = data['L_HEELX'].shape[0],endpoint=True)
            x = data['L_HEELX']
            z = data['L_HEELZ']
            f_interp = interp1d(x,z)    #TODO: may want to add cubic interpolation but kinda too lazy rn
            l_znew = f_interp(l_xnew)
            self.markerless_data[walking_keys[i]]['L_HEELZ_WRT_X'] = l_znew
            l_peaks = signal.find_peaks(-1*l_znew,threshold = -0.05,distance=50)

            if plot:
                t = np.linspace(0,x.shape[0]/100,num=x.shape[0])
                plt.subplot(2,2,1)
                plt.plot(t,data['L_HEELZ'],'b')
                plt.subplot(2,2,3)
                plt.plot(l_xnew,self.markerless_data[walking_keys[i]]['L_HEELZ_WRT_X'].values,'g')
                plt.plot(self.markerless_data[walking_keys[i]]['L_HEELX'].values,self.markerless_data[walking_keys[i]]['L_HEELZ'].values,'y--',alpha=0.8)
                plt.scatter(self.markerless_data[walking_keys[i]]['L_HEELX'].values[l_peaks[0]],self.markerless_data[walking_keys[i]]['L_HEELZ_WRT_X'].values[l_peaks[0]],c='r')
                plt.subplot(2,2,2)
                plt.plot(t,data['R_HEELZ'],'b')
                plt.subplot(2,2,4)
                plt.plot(r_xnew,self.markerless_data[walking_keys[i]]['R_HEELZ_WRT_X'].values,'g')
                plt.plot(self.markerless_data[walking_keys[i]]['R_HEELX'].values,self.markerless_data[walking_keys[i]]['R_HEELZ'].values,'y--')
                plt.scatter(self.markerless_data[walking_keys[i]]['R_HEELX'].values[r_peaks[0]],self.markerless_data[walking_keys[i]]['R_HEELZ_WRT_X'].values[r_peaks[0]],c='r')
                
                plt.suptitle("Interpolated")
                plt.show()
            peakdict.update({walking_keys[i]:[r_peaks[0],l_peaks[0]]})
        return peakdict
            

    def calculate_step_width(self,plot : bool):
        peakdict = self.interpolate_walking(True) # returns {key:[right,left]}
        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        step_width = []
        step_length = []
        for i in range(len(walking_keys)):
            data = self.markerless_data[walking_keys[i]]
            heel = data[['R_HEELX','R_HEELY','R_HEELZ','L_HEELX','L_HEELY','L_HEELZ','L_HEELZ_WRT_X','R_HEELZ_WRT_X']]
            r_peaks = peakdict[walking_keys[i]][0]
            l_peaks = peakdict[walking_keys[i]][1]
            if plot:
                plt.subplot(3,1,2)
                plt.plot(data['R_HEELX'],data['R_HEELY'],'b',data['L_HEELX'],data['L_HEELY'],'g')
                plt.scatter(data['R_HEELX'].values[r_peaks],data['R_HEELY'].values[r_peaks],c='r')
                plt.scatter(data['L_HEELX'].values[l_peaks],data['L_HEELY'].values[l_peaks],c='y')
                plt.legend(['Right Heel','Left Heel','RHS','LHS'])
                plt.ylabel('Lab Y\n(Across)')
                plt.plot([heel['R_HEELX'].values[r_peaks[0]],heel['L_HEELX'].values[l_peaks[0]]], [heel['R_HEELY'].values[r_peaks[0]],heel['L_HEELY'].values[l_peaks[0]]], 'y', linestyle="--")
                xlimmin,xlimmax = plt.xlim()

                plt.subplot(3,1,1)
                plt.plot(data['L_HEELX'],data['L_HEELZ'],'g')
                plt.scatter(data['L_HEELX'].values[l_peaks],data['L_HEELZ'].values[l_peaks],c='y')
                plt.xlim(xlimmin,xlimmax)


                plt.subplot(3,1,3)
                plt.plot(data['R_HEELX'],data['R_HEELZ'],'b')
                plt.scatter(data['R_HEELX'].values[l_peaks],data['R_HEELZ'].values[l_peaks],c='r')
                plt.xlim(xlimmin,xlimmax)
                # f = plt.figure()
                # gs= GridSpec(2,2,figure=f)
                # ax1 = f.add_subplot(gs[0,0])
                # ax2 = f.add_subplot(gs[0,1])
                # ax3 = f.add_subplot(gs[1,:])
                
                # ax1.plot(heel['L_HEELZ_WRT_X'].values)
                # ax1.scatter(l_peaks,heel['L_HEELZ_WRT_X'].values[l_peaks],c='r')
                # ax1.set_ylabel('Lab Z (m)')
                # ax1.set_xlabel('Lab X (cm)')

                # ax2.plot(heel['R_HEELZ_WRT_X'].values,'g')
                # ax2.scatter(r_peaks,heel['R_HEELZ_WRT_X'].values[r_peaks],c='r')
                # ax2.set_ylabel('Lab Z (m)')
                # ax2.set_xlabel('Lab X (m)')
                # ax3.plot(heel['L_HEELX'].values,heel['L_HEELY'].values)
                # ax3.scatter(heel['L_HEELX'].values[l_peaks],heel['L_HEELY'].values[l_peaks],c='r')

                # ax3.plot(heel['R_HEELX'].values,heel['R_HEELY'].values,'g')
                # ax3.scatter(heel['R_HEELX'].values[r_peaks],heel['R_HEELY'].values[r_peaks],c='r')
                # ax3.set_xlabel('Lab X (m)')
                # ax3.set_ylabel('Lab Y (m)')
                # ax3.legend(['Left','Heel Down','Right'])

            #TODO: scaling peaks by x has an issue it seems...maybe use cubic interpolation?
            for _ in range(r_peaks.size+l_peaks.size):
                if r_peaks.size ==0 or l_peaks.size ==0:
                    break
                r_min = np.min(r_peaks)
                l_min = np.min(l_peaks)
                if plot:
                    plt.subplot(3,1,2)
                    plt.plot([heel['R_HEELX'].values[r_peaks[0]],heel['L_HEELX'].values[l_peaks[0]]], [heel['R_HEELY'].values[r_peaks[0]],heel['L_HEELY'].values[l_peaks[0]]], 'pink', linestyle="--")
                step_length.append(np.abs(heel['R_HEELX'].values[r_peaks[0]]-heel['L_HEELX'].values[l_peaks[0]]))
                step_width.append(np.abs(heel['R_HEELY'].values[r_peaks[0]]-heel['L_HEELY'].values[l_peaks[0]]))
                if r_min < l_min:
                    r_peaks = np.delete(r_peaks,0)
                else:
                    l_peaks = np.delete(l_peaks,0)
            if plot:
                print("\nStep Width: %.3f +/- (%.3f)"%(np.mean(step_width),np.std(step_width)))
                print("Step length: %.3f +/- (%.3f)"%(np.mean(step_length),np.std(step_length)))
            if plot:
                plt.suptitle(walking_keys[i]+'\n'+self.id)
                plt.show()
        for j in range(len(step_length)-1,0,-1):
            if step_length[j] > 1000000:
                step_length.pop(i)
                print("Removed")
        self.markerless_step_width = np.array(step_width)/self.height
        self.markerless_step_length = np.array(step_length)/self.height

    def calculate_step_width2(self,plot: bool):
        # trial_names = list(self.markerless_events.keys())
        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        for trial in walking_keys:
            data = self.markerless_data[trial]
            rhs = np.array(self.markerless_events[trial]['RHS'])
            lhs = np.array(self.markerless_events[trial]['LHS'])
            rto = np.array(self.markerless_events[trial]['RTO'])
            lto = np.array(self.markerless_events[trial]['LTO'])
            rhs = np.round(rhs*100).astype(int)
            lhs = np.round(lhs*100).astype(int)
            rto = np.round(rto*100).astype(int)
            lto = np.round(lto*100).astype(int)
            max_ind = np.max(data.index)
            rhs = rhs[:np.argmax(rhs>=max_ind)]
            lhs = lhs[:np.argmax(lhs>=max_ind)]
            rto = rto[:np.argmax(rto>=max_ind)]
            lto = rhs[:np.argmax(lto>=max_ind)]

            if plot:
                plt.subplot(3,1,2)
                plt.plot(data['R_HEELX'],data['R_HEELY'],'b',data['L_HEELX'],data['L_HEELY'],'g')
                plt.scatter(data['R_HEELX'][rhs],data['R_HEELY'][rhs],c='r')
                plt.scatter(data['L_HEELX'][lhs],data['L_HEELY'][lhs],c='y')
                plt.legend(['Right Heel','Left Heel','RHS','LHS'])
                plt.ylabel('Lab Y\n(Across)')
                xminlim,xmaxlim = plt.xlim()

                plt.subplot(3,1,1)
                plt.plot(data['L_HEELX'],data['L_HEELZ'],'g')
                plt.scatter(data['L_HEELX'][lhs],data['L_HEELZ'][lhs],c='y')
                plt.ylabel('Lab Z\n(Vertical)')
                plt.xlim([xminlim,xmaxlim])

                plt.subplot(3,1,3)
                plt.plot(data['R_HEELX'],data['R_HEELZ'],'b')
                plt.scatter(data['R_HEELX'][rhs],data['R_HEELZ'][rhs],c='r')
                plt.ylabel('Lab Z\n(Vertical)')
                plt.xlabel('Lab X\n(Along)')
                plt.xlim([xminlim,xmaxlim])
                print(trial)
                print(rhs)
                print(lhs)
                plt.suptitle(trial + '\n' + self.id+'\nV3D')
                plt.show()
    

    def calculate_step_height(self,plot:bool):
        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        step_height = []
        for i in range(len(walking_keys)):
            data = self.markerless_data[walking_keys[i]]
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
        self.markerless_step_height = np.array(step_height)/self.height

    def calculate_pelvis_pos(self,plot:bool):
        self.markerless_output_data['Pelvis_Jerk'] = np.nan
        self.markerless_output_data['Pelvis_Accel'] = np.nan

        walking_keys = getIndexNames('Walking',self.markerless_task_labels)
        for i in range(len(walking_keys)):
            # the three lines below were added because the different segments get recognized at different points
            data = self.markerless_data[walking_keys[i]]
            data = data.loc[:,['PelvisPosX','PelvisPosY','PelvisPosZ']]
            data = data.dropna(how='all')
            #TODO: may want to add something that cuts out snippets that are recognized for like a second then dissappear
            #z
            # zdata = self.markerless_data[walking_keys[i]]['PelvisPosZ'].values
            zdata = data['PelvisPosZ']
            t = np.linspace(0,zdata.shape[0]/self.markerless_fs,zdata.shape[0])
            zvelocity = np.gradient(zdata,1/self.markerless_fs)
            zaccel = np.gradient(zvelocity,1/self.markerless_fs)
            zjerk = np.gradient(zaccel,1/self.markerless_fs)
            #y
            # ydata = self.markerless_data[walking_keys[i]]['PelvisPosY'].values
            ydata = data['PelvisPosY']
            yvelocity = np.gradient(ydata,1/self.markerless_fs)
            yaccel = np.gradient(yvelocity,1/self.markerless_fs)
            yjerk = np.gradient(yaccel,1/self.markerless_fs)
            #x
            # xdata = self.markerless_data[walking_keys[i]]['PelvisPosX'].values
            xdata = data['PelvisPosX']
            xvelocity = np.gradient(xdata,1/self.markerless_fs)
            xaccel = np.gradient(xvelocity,1/self.markerless_fs)
            xjerk = np.gradient(xaccel,1/self.markerless_fs)

            if plot:
                plt.subplot(4,3,1)
                plt.title('X - Along Lab')
                plt.ylabel('Position (m)')
                plt.plot(t,xdata)

                plt.subplot(4,3,2)
                plt.title('Y - Across Lab')
                plt.plot(t,ydata)

                plt.subplot(4,3,3)
                plt.title('Z - Vertical Lab')
                plt.plot(t,zdata)

                plt.subplot(4,3,4)
                plt.ylabel('Velocity (m/s)')
                plt.plot(t,xvelocity)

                plt.subplot(4,3,5)
                plt.plot(t,yvelocity)

                plt.subplot(4,3,6)
                plt.plot(t,zvelocity)

                plt.subplot(4,3,7)
                plt.ylabel('Acceleration (m^2/s)')
                plt.plot(t,xaccel)

                plt.subplot(4,3,8)
                plt.plot(t,yaccel)

                plt.subplot(4,3,9)
                plt.plot(t,zaccel)

                plt.subplot(4,3,10)  
                plt.ylabel('Jerk (m^3/s)')
                plt.plot(t,xjerk)

                plt.subplot(4,3,11)
                plt.plot(t,yjerk)
                plt.xlabel('Time (s)')
                
                plt.subplot(4,3,12)
                plt.plot(t,zjerk)

                plt.suptitle('Pelvis Measurements vs Time')
            
            #truncate if there is artifact
            # threshold = np.mean(xvelocity) + 3 * np.std(xvelocity)
            threshold = 5
            while np.any(np.abs(xvelocity)>threshold):
                #this is sometimes due to the disappearing and reappearing of the reconstruction in theia...so could look at how
                # the nan data is removed...but this works
                print("\nTruncating due to artifact")
                locs = np.where(np.abs(xvelocity)>threshold)[-1]
                dist = 30
                #z
                zdata = zdata[locs[-1]+dist:]
                zvelocity = zvelocity[locs[-1]+dist:]
                zaccel = zaccel[locs[-1]+dist:]
                zjerk = zjerk[locs[-1]+dist:]
                #y
                ydata = ydata[locs[-1]+dist:]
                yvelocity = yvelocity[locs[-1]+dist:]
                yaccel = yaccel[locs[-1]+dist:]
                yjerk = yjerk[locs[-1]+dist:]
                #x
                xdata = xdata[locs[-1]+dist:]
                xvelocity = xvelocity[locs[-1]+dist:]
                xaccel = xaccel[locs[-1]+dist:]
                xjerk = xjerk[locs[-1]+dist:]
                if plot: 
                    for jd in range(1,13):
                        plt.subplot(4,3,jd)
                        plt.axvline(t[locs[-1]+dist],color='r')

            # https://stackoverflow.com/questions/5613244/root-mean-square-in-numpy-and-complications-of-matrix-and-arrays-of-numpy
            accel_rms = np.sqrt(xaccel.dot(xaccel)/xaccel.size) + np.sqrt(yaccel.dot(yaccel)/yaccel.size) + np.sqrt(zaccel.dot(zaccel)/zaccel.size)
            jerk_rms = np.sqrt(xjerk.dot(xjerk)/xjerk.size) + np.sqrt(yjerk.dot(yjerk)/yjerk.size) + np.sqrt(zjerk.dot(zjerk)/zjerk.size)
            if plot:
                print('\n'+walking_keys[i])
                print("Accel rms: %.3f" % np.mean(accel_rms))
                print("Jerk rms: %.3f" % np.mean(jerk_rms))

            #caculate mean gait speed just using x
            x = data['PelvisPosX'].values
            dist = np.abs(np.min(x))+np.abs(np.max(x))
            time = x.shape[0] / self.markerless_fs
            speed = dist / time
            normalized_accel_rms = accel_rms / speed
            normalized_jerk_rms = jerk_rms / speed
            if plot:
                print("Speed: %.2f"%speed)
                print("Norm Accel rms: %.3f" % np.mean(normalized_accel_rms))
                print("Norm Jerk rms: %.3f" % np.mean(normalized_jerk_rms))
                plt.show()
            #assign output
            self.markerless_output_data.loc[walking_keys[i],'Walking Speed'] = speed
            self.markerless_output_data.loc[walking_keys[i],'Pelvis_Jerk'] = normalized_jerk_rms
            self.markerless_output_data.loc[walking_keys[i],'Pelvis_Accel'] = normalized_accel_rms