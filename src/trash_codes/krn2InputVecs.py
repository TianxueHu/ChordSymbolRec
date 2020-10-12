import os
from music21 import *
from fractions import Fraction
import numpy as np


class KRN2INPUTVEC(object):
    def __init__(self, fp_song):
        self.stream = converter.parse(fp_song)
        # input vector structure:
        # 0-11 : one hot encoding for notes that current playing
        # 12-23 : one hot encoding for notes is onset at this slice
        # 24 : binary encoding for whether the current onset is on beat (strong)
        # 25 : binary encoding for whether the current onset is on beat (weak)
        # 26 : binary encoding for whether the current onset is off beat
        # 27 : measure number (for further processing, delete before feeding into models)
        # 28 : beat position (for further processing, delete before feeding into models)
        self.input_vec_piece = []
        #self.output_vec = []

    
    def define_beat_from_meter(self):
        '''
        Return lists of strong and weak beat given meter
        Output:
            strong - list[int]
            weak - list[int]
        '''
        meter = self.stream.parts[0][0].timeSignature.ratioString
        strong, weak = [], []
        if meter == '2/4':
            strong, weak = [1], [2]
        elif meter == '3/4':
            strong, weak = [1,3], [2]
        elif meter == '4/4':
            strong, weak = [1,3], [2,4]
        elif meter == '3/8': #TBD
            strong, weak = [1,3], [2]
        elif meter == '6/8': #TBD
            strong, weak = [1], [2]
        else:
            raise ValueError("meter not classified, current meter is: ", meter)
        return strong, weak
    

    def proccess_beat(self, beat):
        '''
        Process beat number in case it is a fraction

        Input: 
            beat - string
        Output:
            beat_num - int/float
        '''

        if len(beat) > 1:
            beat_num = float(sum(Fraction(s) for s in beat.split()))
        else:
            beat_num = int(beat)
        return beat_num
    
    def get_beat_vector(self, beat):
        '''
        One hot encoding for beat position: on strong beat, on weak beat, off beat

        Input: 
            beat - string
        Output:
            beat_vec - list[int] of 3 elements
        '''
        beat_num = self.proccess_beat(beat)
        #print(self.stream.parts[0][0].timeSignature.ratioString)
        #print(beat_num)
        strong, weak = self.define_beat_from_meter()
        if beat_num in strong:
            beat_vec = [1,0,0]
        elif beat_num in weak:
            beat_vec = [0,1,0]
        else:
            beat_vec = [0,0,1]
        #print(beat_vec)
        return beat_vec 


    
    def is_rest(self, cur_onset_notes, prev_onset_notes):
        '''
        Due to eliminating NCT in the pre-processing step, there might be 
        some slices only containing rests, use this function to get whether
        the current slice is a "rest slice", if yes, do not consider it as an
        input vector 

        Input: 
            prev_onset_notes - list[int]
            cur_onset_nots - list[int]
        Output:
            is_rest - boolean
        '''
        if len(cur_onset_notes) < len(prev_onset_notes) and set(cur_onset_notes).issubset(prev_onset_notes):
            return True
        else:
            return False 
    
    
    def get_onset_notes(self, cur_onset_notes, prev_onset_notes):
        '''
        Get notes that attack on this slice

        Input: 
            prev_onset_notes - list[int]
            cur_onset_nots - list[int]
        Output:
            onset_notes - list[int]
        '''

        onset_notes = [x for x in cur_onset_notes if x not in prev_onset_notes]
        return onset_notes

    
    def to_12d_vec(self, midi_note_list):
        '''
        one-hot encoding on 12 dimension (12 semitones) with input of a list in midi number

        Input: 
            note_list - list[int]
        Output:
            output_vec - list[int] of 12 elements
        '''
        output_vec = [0]*12
        for midi in midi_note_list:
            n = midi%12
            if output_vec[n] == 0:
                output_vec[n] = 1
        return output_vec

    
    def gen_input_vec_onset_slice(self):
        '''
        Generates input vector for each onset slice in this piece
        '''
        #using Chordify to get onset slices
        chord = self.stream.chordify()

        #for each onset slice
        prev_notes = []
        for thisChord in chord.recurse().getElementsByClass('Chord'):
            cur_input_vec = []
            cur_notes = []
            for pitch in thisChord.pitches:
                cur_notes.append(int(pitch.midi))
            #print("cur note", cur_notes)
            
            cur_is_rest = self.is_rest(cur_notes, prev_notes)    #check if is a "rest slice"
            if cur_is_rest:
                continue
            
            onset_notes = self.get_onset_notes(cur_notes, prev_notes) #get onset notes
            #print("os notes", onset_notes)
            prev_notes = cur_notes

            cur_vec = self.to_12d_vec(cur_notes)   #get vector for current playing notes
            onset_vec = self.to_12d_vec(onset_notes)    #get vector for current onset notes
            #print("cur_vec",cur_vec, "onset_vec", onset_vec)
            cur_input_vec.extend(cur_vec)
            cur_input_vec.extend(onset_vec)

            measure_num = thisChord.measureNumber    #current measure 
            beat = thisChord.beatStr     # current beat in the measure
            beat_vec = self.get_beat_vector(beat)
            cur_input_vec.extend(beat_vec)
            cur_input_vec.append(int(measure_num))
            cur_input_vec.append(self.proccess_beat(beat))
            print(cur_input_vec)

            self.input_vec_piece.append(cur_input_vec)
            



if __name__ == "__main__":
    # ----------------test script---------------------------------
    script_dir = os.path.dirname(__file__)
    # filepath of original score
    score_rel_path = "../datasets/bhchorale/bach_final_scores/chor002_score_reduced4.krn"
    scorepath = os.path.join(script_dir, score_rel_path)

    #boo = if_rest(self, [60,62,64], [60,62,64])

    input_vec = KRN2INPUTVEC(scorepath)
    input_vec.gen_input_vec_onset_slice()
    print(np.array(input_vec.input_vec_piece).shape)
    #lrc_object.save2xml(save_dir)
