import os
import pandas as pd
import numpy as np
from utils import to_21d_vec, get_beat_vector, specialChords
import re
from harmalysis import inputRN2Chord
import pickle
#import sys
#sys.path.insert(0, './harmalysis')
#from inputRN2Chord import inputRN


class KRN2VEC(object): 
    def __init__(self, fp_song):
        # read song to dataframe
        self.df = pd.read_csv(fp_song, sep="\t", header=None)
        self.piece_output = []
    
    
    def add_header(self, collection):
        if collection == "Haydn_REDUCE":
            self.df.columns = ["harm", "voice4", "voice3", "voice2", "voice1", "index", "beat", "measure", "key", "meter" ]
                
        elif collection == "Bach_REDUCE":
            self.df.columns = ["harm", "root", "voice4", "voice3", "voice2", "voice1", "index", "beat", "measure", "key", "meter" ]

        elif collection == "Haydn_ORG":
            self.df.columns = ["harm", "voice4", "voice3", "voice2", "voice1", "beat", "measure", "key", "meter" ]
        
        elif collection == "Bach_ORG":
            self.df.columns = ["harm", "root", "voice4", "voice3", "voice2", "voice1", "beat", "measure", "key", "meter" ]
        
        elif collection == "Sears_ORG":
            self.df.columns = ["voice4", "voice3", "voice2", "voice1", "harm", "beat", "measure", "key", "meter" ]

        elif collection == "Sears_REDUCE":
            self.df.columns = ["voice4", "voice3", "voice2", "voice1", "harm", "index", "beat", "measure", "key", "meter" ]

        # process dataframe
        self.df = self.df[~self.df['beat'].astype(str).str.startswith(('=','.','*'))]
    

    def krn2vec_ffnn_21(self, collection):
        self.add_header(collection)
    
        prev_note_list = []
        for index, row in self.df.iterrows():
            # for this onset slice 
            this_onset_vec = []
        
            ######################################### Process notes and onsets ########################################
            voices = ["voice4","voice3","voice2","voice1"]
            cur_note_tmp = []
            for v in voices:
                this_note = ''.join(row[[v]].values)
                if ' ' in this_note: #for a part has multiple voices
                    this_note_list = this_note.split(" ")
                    cur_note_tmp.extend(this_note_list)
                else:
                    cur_note_tmp.append(this_note)
            
            cur_note_list = []
            for n in cur_note_tmp:
                note_name = re.sub('[^a-gA-G#-]+', '', n)
                if note_name:
                    cur_note_list.append(note_name)
                    
            cur_onset_list = list(set(cur_note_list) - set(prev_note_list))
            prev_note_list = cur_note_list
            
            note_vec = to_21d_vec(cur_note_list)
            onset_vec = to_21d_vec(cur_onset_list)
            this_onset_vec.extend(note_vec)
            this_onset_vec.extend(onset_vec)
            
            ######################################### Process beat position ########################################
            beat_pos = ''.join(row[["beat"]].values)
            meter = ''.join(row[["meter"]].values)
            meter = meter.replace("M", "")
            beat_vec = get_beat_vector(beat_pos, meter)
            this_onset_vec.extend(beat_vec)
            
            ######################################### Process chord label ########################################
            harm = ''.join(row[["harm"]].values)
            harm = re.sub('[();]', '', harm) # process string
            harm = specialChords(harm)
            key = ''.join(row[["key"]].values)
            key_harm = key + ":" + harm
            try:
                chord_label = inputRN2Chord.inputRN(key_harm)["Chord label"]
            except:
                continue # pass this onset slice if RN is not recognizable
            this_onset_vec.append(str(chord_label))
            self.piece_output.append(this_onset_vec)



if __name__ == "__main__":
    script_dir = os.getcwd()
    SCORE_COLLECTION_REL_PATH = "datasets/Sears_Corpus/sears_org_score_for_vec/"
    COLLECTION = "Sears_ORG"
    collection_path = os.path.join(script_dir, SCORE_COLLECTION_REL_PATH)

    collection_list = []
    bad_files = []
    for subdir, dirs, files in os.walk(collection_path):
        num_files = len(files)
        for idx, file in enumerate(files):
            print('Processing ', idx, " of ", num_files, " files.")
            ext = os.path.splitext(file)[-1].lower()
            if ext == ".krn":
                scorepath = os.path.join(subdir, file)
                print (os.path.join(subdir, file))
                try:
                    vec = KRN2VEC(scorepath)
                    vec.krn2vec_ffnn_21(COLLECTION)
                    collection_list.append(vec.piece_output)
                except:
                    bad_files.append(file)
                    pass
    
    #print(np.array(collection_list).shape)
    print("Bad files:", bad_files)

    with open('Sears_org_vectors_ffnn_21enc.pkl', 'wb') as f:
        pickle.dump(collection_list, f)
    print("Pickle vector list saved!")
