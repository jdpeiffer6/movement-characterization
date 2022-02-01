import numpy
import os
import json
import pandas as pd
import glob

class DataObject:
    def __init__(self, path):
        print("loading data subject %s" % path[-3:])
        self.path = path
        self.id = path[-3:]
        sessions = [x[0] for x in os.walk(path)]
        del sessions[0]
        print('found %d sessions for subject %s' % (len(sessions), self.id))
        self.sessions = sessions

        self.load_markerless()
        self.load_markers()

    def load_markerless(self):
        #load data
        # this uses the fact that all markerless will be 001 and all marker will be 002
        markerless_folders = self.sessions[0:len(self.sessions):2]

        # load markerless
        self.markerless_data = {}
        with open(glob.glob(markerless_folders[0]+'/*.json')[0]) as f:
            markerless_data_json = json.load(f)
            self.markerless_json_path = glob.glob(markerless_folders[0]+'/*.json')[0]

        for i in range(len(markerless_data_json['Visual3D'])):
            data = markerless_data_json['Visual3D'][i]
            trial_name = data['filename']
            trial_name = trial_name.split('/')[-1][0:-4]
            if trial_name not in self.markerless_data.keys():
                self.markerless_data[trial_name] = pd.DataFrame()
                print(trial_name)
            for j in range(len(data['signal'])):
                self.markerless_data[trial_name][data['name']+data['signal'][j]['component']] = data['signal'][j]['data']
            
        # load task labels
        self.markerless_task_labels = pd.read_csv(glob.glob(markerless_folders[0]+'/*.csv')[0])
        #TODO: more sorting using this task labels

    def load_markers(self):
        marker_folders = self.sessions[1:len(self.sessions):2]
        #TODO: this only adds one folder
        marker_files = glob.glob(marker_folders[0] + '/*.tsv')

