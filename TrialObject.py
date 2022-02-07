import os
import json
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mpld3
from scipy import signal
from scipy.interpolate import interp1d
import numpy as np

class TrialObject:
    def __init__(self, path):
        # this path will be the path to each subject's folder and the date format for their session
        # for instance "...\\002\\2022-01-28"
        print("loading data subject %s for date %s\n" % (path[-14:-11], path[-10:]))
        #patient metrics
        self.height = 1.7 # meters

        #variable assignments
        self.trial_path = path
        self.id = path[-14:-11]
        self.date = path[-10:]
        # this uses the fact that all markerless will be 001 and all marker will be 002
        self.sessions = [path + '_001', path + '_002']
        for session in self.sessions:
            if not os.path.isdir(session):
                raise Exception("could not find both paths for subject %s"%self.id)
        self.markerless_folder = self.sessions[0]
        self.marker_folder = self.sessions[1]
        self.markerless_fs = 80
        self.marker_fs = 100

        self.step_width = []   #TODO: delete these
        self.step_length = []
        self.step_height = []

        self.load_markerless()
        self.load_markers()
        # self.create_walking_plots()
        self.analyze_walking()
        # self.create_tandem_plots()

    def load_markerless(self):
        self.markerless_data = {}
        self.markerless_events = {}
        with open(glob.glob(self.markerless_folder+'/*.json')[0]) as f:
            markerless_data_json = json.load(f)
            self.markerless_json_path = glob.glob(self.markerless_folder+'/*.json')[0]

        for i in range(len(markerless_data_json['Visual3D'])):
            data = markerless_data_json['Visual3D'][i]
            trial_name = data['filename']
            trial_name = trial_name.split('/')[-1][0:-4]
            if trial_name not in self.markerless_data.keys():
                self.markerless_data[trial_name] = pd.DataFrame()
                self.markerless_events[trial_name] = {}
                print(trial_name)
            for j in range(len(data['signal'])):
                if data['name'] in ['LHS','LTO','RHS','RTO']:
                    self.markerless_events[trial_name].update({data['name']:data['signal'][j]['data']})
                    #append to different container
                    a=5
                else:
                    self.markerless_data[trial_name][data['name']+data['signal'][j]['component']] = data['signal'][j]['data']
        
        for i in self.markerless_data.keys():
            self.markerless_data[i].dropna(how='all',inplace=True)

        # load task labels
        self.markerless_task_labels = pd.read_csv(glob.glob(self.markerless_folder+'/*.csv')[0])
        #TODO: more sorting using this task labels

    def load_markers(self):
        self.marker_files = glob.glob(self.marker_folder + '/*.tsv')
        self.marker_data = {}
        for i in self.marker_files:
            name = (i.split('\\')[-1]).split('.')[0]
            data = pd.read_csv(i,sep='\t',skiprows=11)
            self.marker_data[name] = data
            print(name)
        self.marker_task_labels = pd.read_csv(glob.glob(self.marker_folder+'/*.csv')[0])

    def create_walking_plots(self):
        self.colors = ['#130AF1','#17ACE8','#1BD9DE']
        plotlist = []
        plotlabels=[]
        walking_keys = self.markerless_task_labels[self.markerless_task_labels['Task'] == 'Walking']['Name']
        f_pelvis,ax_pelvis = plt.subplots(len(walking_keys),2,figsize=(20,5))
        for i in range(len(walking_keys)):
            data = self.markerless_data[walking_keys[i]]
            t=data.index.to_list()
            t0=t[0]
            t = [(t[h] - t0)/self.markerless_fs for h in range(len(t)) ]
            ax_pelvis[i][0].plot(data['Pelvis_WRT_LabX'],data['Pelvis_WRT_LabY'])
            ax_pelvis[i][1].plot(t,data['Pelvis_WRT_LabZ'])
            ax_pelvis[i][0].set_ylim(bottom = -1,top=1)
            #TODO: Clean up this figure
        f_pelvis.set_figwidth(15)
        ax_pelvis[0][0].set_title("Pelvis X/Y path across lab")
        ax_pelvis[0][1].set_title("Pelvis height vs Time")
        f_pelvis.suptitle("Pelvis Position")
        f_pelvis.tight_layout()
        plotlist.append(f_pelvis)
        plotlabels.append('Pelvis Position')

        f_angles,ax_angles = plt.subplots(len(walking_keys),6,figsize = (20,7))
        f_angles.tight_layout()
        for i in range(len(walking_keys)):
            data = self.markerless_data[walking_keys[i]]
            t=data.index.to_list()
            t0=t[0]
            t = [(t[h] - t0)/self.markerless_fs for h in range(len(t)) ]
            ax_angles[i][0].plot(t,data['Left Ankle AnglesX'],color=self.colors[0])
            ax_angles[i][1].plot(t,data['Left Knee AnglesX'],color=self.colors[1])
            ax_angles[i][2].plot(t,data['Left Hip AnglesX'],color=self.colors[2])
            ax_angles[i][3].plot(t,data['Right Ankle AnglesX'],color=self.colors[0])
            ax_angles[i][4].plot(t,data['Right Knee AnglesX'],color=self.colors[1])
            ax_angles[i][5].plot(t,data['Right Knee AnglesX'],color=self.colors[2])
            #TODO: add right hip angles

        ax_angles[0][0].set_title("Left Ankle Angle")
        ax_angles[0][1].set_title("Left Knee Angle")
        ax_angles[0][2].set_title("Left Hip Angle")
        ax_angles[0][3].set_title("Right Ankle Angle")
        ax_angles[0][4].set_title("Right Knee Angle")
        ax_angles[0][5].set_title("Right Knee Angle")
        f_angles.set_figwidth(15)
        plotlist.append(f_angles)
        plotlabels.append('Joint Angles')

        f_feetz,ax_feetz = plt.subplots(len(walking_keys),2,figsize = (20,6))
        f_feetz.tight_layout()
        for i in range(len(walking_keys)):
            data = self.markerless_data[walking_keys[i]]
            t=data.index.to_list()
            t0=t[0]
            t = [(t[h] - t0)/self.markerless_fs for h in range(len(t)) ]
            ax_feetz[i][0].plot(t,data['L_HEELZ'])
            ax_feetz[i][1].plot(t,data['R_HEELZ'])
            ax_feetz[i][1].plot(t,data['RTOES_DISTALZ'],'r')
            ax_feetz[i][1].legend(['Heel','Toes'])

        ax_feetz[0][0].set_title("Left Heel Height")
        ax_feetz[0][1].set_title("Right Foot Height",fontweight='bold')
        plotlist.append(f_feetz)
        plotlabels.append('Foot Height')

        f_feetxy,ax_feetxy = plt.subplots(len(walking_keys),1,figsize=(20,6))        
        f_feetxy.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
        f_feetxy.tight_layout()
        for i in range(len(walking_keys)):
            data = self.markerless_data[walking_keys[i]]
            ax_feetxy[i].plot(data['R_HEELX'],data['R_HEELY'])
            ax_feetxy[i].plot(data['L_HEELX'],data['L_HEELY'],color='r')
            ax_feetxy[i].legend(['R','L'])
        ax_feetxy[0].set_title("Foot XY Position (top down lab view)")
        plotlist.append(f_feetxy) 
        plotlabels.append('Step locations')

        if not os.path.isdir(self.trial_path + '_output'):
            os.mkdir(self.trial_path + '_output')

        HTML_FILE = open(self.trial_path+'_output\\walking.html',"w")
        HTML_FILE.write("""<html>
        <head>
            <title>Walking</title>
            <style>
                h1{text-align: center;}
            </style>
        </head>
        <body>
            <h1>Walking\n"""+self.id+"\n"+self.date+"""</h1>
        </body>
            </html>""")
        for i in range(len(plotlist)):
            HTML_FILE.write('<h2>'+plotlabels[i]+'</h2>')
            html_str = mpld3.fig_to_html(plotlist[i])
            HTML_FILE.write(html_str)
        HTML_FILE.close()

        print("Completed walking html export")

    def analyze_walking(self):
        self.calculate_step_height()
        self.calculate_step_width()
        # self.calculate_step_width2()
        

    def interpolate_walking(self):
        walking_keys = self.markerless_task_labels[self.markerless_task_labels['Task'] == 'Walking']['Name']
        peakdict = {}
        for i in range(len(walking_keys)):
            data = self.markerless_data[walking_keys[i]]
             #interpolate that sh(t)
            # right leg
            xmin = min(data['R_HEELX'])
            xmax = max(data['R_HEELX'])
            xnew = np.linspace(xmin,xmax,num = data['R_HEELX'].shape[0],endpoint=True)
            x = data['R_HEELX']
            z = data['R_HEELZ']
            f_interp = interp1d(x,z)
            r_znew = f_interp(xnew)
            self.markerless_data[walking_keys[i]]['R_HEELZ_WRT_X'] = r_znew
            r_peaks = signal.find_peaks(-1*r_znew,threshold = -0.05,distance=50)

            # left leg
            xmin = min(data['L_HEELX'])
            xmax = max(data['L_HEELX'])
            xnew = np.linspace(xmin,xmax,num = data['L_HEELX'].shape[0],endpoint=True)
            x = data['L_HEELX']
            z = data['L_HEELZ']
            f_interp = interp1d(x,z)    #TODO: may want to add cubic interpolation but kinda too lazy rn
            l_znew = f_interp(xnew)
            self.markerless_data[walking_keys[i]]['L_HEELZ_WRT_X'] = l_znew
            l_peaks = signal.find_peaks(-1*l_znew,threshold = -0.05,distance=50)

            # t = np.linspace(0,x.shape[0]/100,num=x.shape[0])
            # plt.subplot(2,2,1)
            # plt.plot(t,data['L_HEELZ'],'b')
            # plt.subplot(2,2,3)
            # plt.plot(self.markerless_data[walking_keys[i]]['L_HEELZ_WRT_X'].values,'g')
            # plt.scatter(l_peaks[0],self.markerless_data[walking_keys[i]]['L_HEELZ_WRT_X'].values[l_peaks[0]],c='r')
            # plt.subplot(2,2,2)
            # plt.plot(t,data['R_HEELZ'],'b')
            # plt.subplot(2,2,4)
            # plt.plot(self.markerless_data[walking_keys[i]]['R_HEELZ_WRT_X'].values,'g')
            # plt.plot(self.markerless_data[walking_keys[i]]['R_HEELX'].values,self.markerless_data[walking_keys[i]]['R_HEELZ'].values,'y--')
            # plt.scatter(r_peaks[0],self.markerless_data[walking_keys[i]]['R_HEELZ_WRT_X'].values[r_peaks[0]],c='r')
            
            # plt.suptitle("Interpolated")
            # plt.show()
            peakdict.update({walking_keys[i]:[r_peaks[0],l_peaks[0]]})
        return peakdict
            

    def calculate_step_width(self):
        peakdict = self.interpolate_walking() # returns {key:[right,left]}
        walking_keys = self.markerless_task_labels[self.markerless_task_labels['Task'] == 'Walking']['Name']
        for i in range(len(walking_keys)):
            data = self.markerless_data[walking_keys[i]]
            heel = data[['R_HEELX','R_HEELY','R_HEELZ','L_HEELX','L_HEELY','L_HEELZ','L_HEELZ_WRT_X','R_HEELZ_WRT_X']]
            r_peaks = peakdict[walking_keys[i]][0]
            l_peaks = peakdict[walking_keys[i]][1]
            f = plt.figure()
            gs= GridSpec(2,2,figure=f)
            ax1 = f.add_subplot(gs[0,0])
            ax2 = f.add_subplot(gs[0,1])
            ax3 = f.add_subplot(gs[1,:])
            
            ax1.plot(heel['L_HEELZ_WRT_X'].values)
            ax1.scatter(l_peaks,heel['L_HEELZ_WRT_X'].values[l_peaks],c='r')
            ax1.set_ylabel('Lab Z (m)')
            ax1.set_xlabel('Lab X (cm)')

            ax2.plot(heel['R_HEELZ_WRT_X'].values,'g')
            ax2.scatter(r_peaks,heel['R_HEELZ_WRT_X'].values[r_peaks],c='r')
            ax2.set_ylabel('Lab Z (m)')
            ax2.set_xlabel('Lab X (m)')
            ax3.plot(heel['L_HEELX'].values,heel['L_HEELY'].values)
            ax3.scatter(heel['L_HEELX'].values[l_peaks],heel['L_HEELY'].values[l_peaks],c='r')

            ax3.plot(heel['R_HEELX'].values,heel['R_HEELY'].values,'g')
            ax3.scatter(heel['R_HEELX'].values[r_peaks],heel['R_HEELY'].values[r_peaks],c='r')
            ax3.set_xlabel('Lab X (m)')
            ax3.set_ylabel('Lab Y (m)')
            ax3.legend(['Left','Heel Down','Right'])

            #TODO: scaling peaks by x has an issue it seems...maybe use cubic interpolation?
            for _ in range(r_peaks.size+l_peaks.size):
                if r_peaks.size ==0 or l_peaks.size ==0:
                    break
                r_min = np.min(r_peaks)
                l_min = np.min(l_peaks)
                plt.plot([heel['R_HEELX'].values[r_peaks[0]],heel['L_HEELX'].values[l_peaks[0]]], [heel['R_HEELY'].values[r_peaks[0]],heel['L_HEELY'].values[l_peaks[0]]], 'yo', linestyle="--")
                self.step_length.append(np.abs(heel['R_HEELX'].values[r_peaks[0]]-heel['L_HEELX'].values[l_peaks[0]]))
                self.step_width.append(np.abs(heel['R_HEELY'].values[r_peaks[0]]-heel['L_HEELY'].values[l_peaks[0]]))
                if r_min < l_min:
                    r_peaks = np.delete(r_peaks,0)
                else:
                    l_peaks = np.delete(l_peaks,0)
            print("Step Width: %.3f +/- (%.3f)"%(np.mean(self.step_width),np.std(self.step_width)))
            print("Step length: %.3f +/- (%.3f)"%(np.mean(self.step_length),np.std(self.step_length)))
            plt.suptitle(walking_keys[i])
            plt.show()
            del f,ax1,ax2,ax3
            plt.close()

    def calculate_step_width2(self):
        trial_names = list(self.markerless_events.keys())
        for trial in trial_names:
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

            plt.plot(data['R_HEELX'],data['R_HEELY'],'b',data['L_HEELX'],data['L_HEELY'],'g')
            plt.scatter(data['R_HEELX'][rto],data['R_HEELY'][rto],c='r')
            plt.scatter(data['L_HEELX'][lto],data['L_HEELY'][lto],c='y')
            print(trial)
            print(rhs)
            print(lhs)
            plt.show()
    

    def calculate_step_height(self):
        walking_keys = self.markerless_task_labels[self.markerless_task_labels['Task'] == 'Walking']['Name']
        for i in range(len(walking_keys)):
            data = self.markerless_data[walking_keys[i]]
            # L Heel
            plt.subplot(1,2,1)
            plt.plot(data['L_HEELZ'].values)
            l_peaks = signal.find_peaks(data['L_HEELZ'].values,height=0.1)[0]
            plt.scatter(l_peaks,data['L_HEELZ'].values[l_peaks],c='r')

            #R Heel
            plt.subplot(1,2,2)
            plt.plot(data['R_HEELZ'].values)
            r_peaks = signal.find_peaks(data['R_HEELZ'].values,height=0.1)[0]
            plt.scatter(r_peaks,data['R_HEELZ'].values[r_peaks],c='r')
            plt.suptitle(walking_keys[i])
            
            for j in l_peaks:
                self.step_height.append(data['L_HEELZ'].values[j])
            for j in r_peaks:
                self.step_height.append(data['R_HEELZ'].values[j])
            print("Step height: %.3f +/- (%.3f)"%(np.mean(self.step_height),np.std(self.step_height)))
            plt.show()


    def create_tandem_plots(self):
        self.colors = ['#130AF1','#17ACE8','#1BD9DE']
        plotlist = []
        plotlabels=[]
        tandem_keys = self.markerless_task_labels[self.markerless_task_labels['Task'] == 'Tandem']['Name'].to_list()
        walking_keys = self.markerless_task_labels[self.markerless_task_labels['Task'] == 'Walking']['Name']
        f_pelvis,ax_pelvis = plt.subplots(len(tandem_keys),2,figsize=(20,5))
        for i in range(len(tandem_keys)):
            data = self.markerless_data[tandem_keys[i]]
            t=data.index.to_list()
            t0=t[0]
            t = [(t[h] - t0)/self.markerless_fs for h in range(len(t)) ]
            ax_pelvis[i][0].plot(data['Pelvis_WRT_LabX'],data['Pelvis_WRT_LabY'])
            ax_pelvis[i][1].plot(t,data['Pelvis_WRT_LabZ'])
            ax_pelvis[i][0].set_ylim(bottom = -1,top=1)
            #TODO: Clean up this figure
        f_pelvis.set_figwidth(15)
        ax_pelvis[0][0].set_title("Pelvis X/Y path across lab")
        ax_pelvis[0][1].set_title("Pelvis height vs Time")
        f_pelvis.suptitle("Pelvis Position")
        f_pelvis.tight_layout()
        plotlist.append(f_pelvis)
        plotlabels.append('Pelvis Position')

        f_angles,ax_angles = plt.subplots(len(tandem_keys),6,figsize = (20,7))
        f_angles.tight_layout()
        for i in range(len(tandem_keys)):
            data = self.markerless_data[tandem_keys[i]]
            t=data.index.to_list()
            t0=t[0]
            t = [(t[h] - t0)/self.markerless_fs for h in range(len(t)) ]
            ax_angles[i][0].plot(t,data['Left Ankle AnglesX'],color=self.colors[0])
            ax_angles[i][1].plot(t,data['Left Knee AnglesX'],color=self.colors[1])
            ax_angles[i][2].plot(t,data['Left Hip AnglesX'],color=self.colors[2])
            ax_angles[i][3].plot(t,data['Right Ankle AnglesX'],color=self.colors[0])
            ax_angles[i][4].plot(t,data['Right Knee AnglesX'],color=self.colors[1])
            ax_angles[i][5].plot(t,data['Right Knee AnglesX'],color=self.colors[2])
            #TODO: add right hip angles

        ax_angles[0][0].set_title("Left Ankle Angle")
        ax_angles[0][1].set_title("Left Knee Angle")
        ax_angles[0][2].set_title("Left Hip Angle")
        ax_angles[0][3].set_title("Right Ankle Angle")
        ax_angles[0][4].set_title("Right Knee Angle")
        ax_angles[0][5].set_title("Right Knee Angle")
        f_angles.set_figwidth(15)
        plotlist.append(f_angles)
        plotlabels.append('Joint Angles')

        f_feetz,ax_feetz = plt.subplots(len(tandem_keys),2,figsize = (20,6))
        f_feetz.tight_layout()
        for i in range(len(tandem_keys)):
            data = self.markerless_data[tandem_keys[i]]
            t=data.index.to_list()
            t0=t[0]
            t = [(t[h] - t0)/self.markerless_fs for h in range(len(t)) ]
            ax_feetz[i][0].plot(t,data['L_HEELZ'])
            ax_feetz[i][1].plot(t,data['R_HEELZ'])
            ax_feetz[i][1].plot(t,data['RTOES_DISTALZ'],'r')
            ax_feetz[i][1].legend(['Heel','Toes'])

        ax_feetz[0][0].set_title("Left Heel Height")
        ax_feetz[0][1].set_title("Right Foot Height",fontweight='bold')
        plotlist.append(f_feetz)
        plotlabels.append('Foot Height')

        f_feetxy,ax_feetxy = plt.subplots(len(tandem_keys),1,figsize=(20,6))        
        f_feetxy.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
        f_feetxy.tight_layout()
        for i in range(len(tandem_keys)):
            data = self.markerless_data[tandem_keys[i]]
            ax_feetxy[i].plot(data['R_HEELX'],data['R_HEELY'])
            ax_feetxy[i].plot(data['L_HEELX'],data['L_HEELY'],color='r')
            ax_feetxy[i].legend(['R','L'])
        ax_feetxy[0].set_title("Foot XY Position (top down lab view)")
        plotlist.append(f_feetxy) 
        plotlabels.append('Step locations')

        if not os.path.isdir(self.trial_path + '_output'):
            os.mkdir(self.trial_path + '_output')

        HTML_FILE = open(self.trial_path+'_output\\tandem.html',"w")
        HTML_FILE.write("""<html>
        <head>
            <title>Walking</title>
            <style>
                h1{text-align: center;}
            </style>
        </head>
        <body>
            <h1>Tandem\n"""+self.id+"\n"+self.date+"""</h1>
        </body>
            </html>""")
        for i in range(len(plotlist)):
            HTML_FILE.write('<h2>'+plotlabels[i]+'</h2>')
            html_str = mpld3.fig_to_html(plotlist[i])
            HTML_FILE.write(html_str)
        HTML_FILE.close()

        print("Completed tandem html export")