import os
import json
import pandas as pd
import glob
import matplotlib.pyplot as plt
import mpld3

class TrialObject:
    def __init__(self, path):
        # this path will be the path to each subject's folder and the date format for their session
        # for instance "...\\002\\2022-01-28"
        print("loading data subject %s for date %s\n" % (path[-14:-11], path[-10:]))
        
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

        self.load_markerless()
        self.load_markers()
        #self.create_walking_plots()
        self.create_tandem_plots()

    def load_markerless(self):
        self.markerless_data = {}
        with open(glob.glob(self.markerless_folder+'/*.json')[0]) as f:
            markerless_data_json = json.load(f)
            self.markerless_json_path = glob.glob(self.markerless_folder+'/*.json')[0]

        for i in range(len(markerless_data_json['Visual3D'])):
            data = markerless_data_json['Visual3D'][i]
            trial_name = data['filename']
            trial_name = trial_name.split('/')[-1][0:-4]
            if trial_name not in self.markerless_data.keys():
                self.markerless_data[trial_name] = pd.DataFrame()
                print(trial_name)
            for j in range(len(data['signal'])):
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