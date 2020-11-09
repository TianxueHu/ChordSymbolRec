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
            #print(harm)
            harm = specialChords(harm)
            key = ''.join(row[["key"]].values)
            key_harm = key + ":" + harm
            try:
                chord_label = inputRN2Chord.inputRN(key_harm)["Chord label"]
            except:
                continue # pass this onset slice if RN is not recognizable
            this_onset_vec.append(str(chord_label))
            self.piece_output.append(this_onset_vec)


    def get_mea_num(self, measure):
        if measure == '.':
            measure = 1
        else:
            measure = int(re.findall("\d+", measure)[0])
        return measure


    def get_window_vecs(self, all_vecs, head_idx, this_mea, window_size):
        window_vec = []
        window_onsets_list = []
        window_chords_list = []
        
        next_tail_mea = this_mea
        tail_idx = head_idx
        while next_tail_mea <= this_mea + window_size - 1:
            #this_tail_mea = self.get_mea_num(all_vecs[tail_idx][-1])
            this_onset = all_vecs[tail_idx][:-2]
            window_onsets_list.append(this_onset)
            
            this_chord = all_vecs[tail_idx][-2]
            window_chords_list.append(this_chord)
            '''
            if not window_chords_list:
                window_chords_list.append(this_chord)
            elif window_chords_list and this_chord != window_chords_list[-1]:
                window_chords_list.append(this_chord)
            '''
            
            if tail_idx+1 < len(all_vecs):
                tail_idx += 1
            #print(tail_idx)
            next_tail_mea = self.get_mea_num(all_vecs[tail_idx][-1])
            #print(next_tail_mea)
        
        window_vec.append(window_onsets_list)
        window_vec.append(window_chords_list)
        return window_vec


    def get_all_vec_with_mea(self, collection):
        self.add_header(collection)
        prev_note_list = []
        all_vecs = []
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
            this_onset_vec.append(''.join(row[['measure']].values))
            #print(this_onset_vec)
            all_vecs.append(this_onset_vec)
        return all_vecs

 
    def krn2vec_s2s_21(self, collection, mea_window_size):
        all_vecs = self.get_all_vec_with_mea(collection)
        max_mea = self.get_mea_num(''.join(self.df['measure'].iloc[-1]))
        pre_mea = 1
        #all_windows_vec = []  # cur_window_list = [[window_onsets_list][window_chords_list]]
        for idx, vec in enumerate(all_vecs):
            this_mea = self.get_mea_num(vec[-1])

            if this_mea >= max_mea - mea_window_size + 1:
                break
            
            if this_mea != pre_mea:
                #new measure -> proceed to a new window
                this_window_vec = self.get_window_vecs(all_vecs, idx, this_mea, mea_window_size)
                self.piece_output.append(this_window_vec)
                #print(this_mea, this_window_vec)
                pre_mea = this_mea
                
            

        #self.piece_output.append(all_windows_vec)


if __name__ == "__main__":
    script_dir = os.getcwd()
    SCORE_COLLECTION_REL_PATH = "datasets/haydn_op20_harm/haydn_reduced_score_for_vec/"
    COLLECTION = "Haydn_REDUCE"
    WINDOW_SIZE = 4

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
                    #vec.krn2vec_ffnn_21(COLLECTION)
                    vec.krn2vec_s2s_21(COLLECTION, WINDOW_SIZE)
                    collection_list.append(vec.piece_output)
                except:
                    bad_files.append(file)
                    pass
    
    print("Bad files:", bad_files)

    with open('haydn_reduce_vectors_s2s_21enc_4meaWindow_ditto.pkl', 'wb') as f:
        pickle.dump(collection_list, f)
    print("Pickle vector list saved!")
