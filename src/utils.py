from fractions import Fraction
import re

def define_beat_from_meter(meter):
        '''
        Return lists of strong and weak beat given meter
        Output:
            strong - list[int]
            weak - list[int]
        '''
        strong, weak = [], []
        if meter == '2/4':
            strong, weak = [1], [2]
        elif meter == '3/4':
            strong, weak = [1,3], [2]
        elif meter == '4/4':
            strong, weak = [1,3], [2,4]
        elif meter == '3/8': # krn beat pos in 1, 1.5, 2
            strong, weak = [1], [1.5,2]
        elif meter == '6/8': # krn beat pos in 1, 2, 3
            strong, weak = [1], [2.5]
        else:
            raise ValueError("meter not classified, current meter is: ", meter)
        return strong, weak
    

def get_beat_vector(beat_pos, meter):
    '''
    One hot encoding for beat position: on strong beat, on weak beat, off beat

    Input: 
        beat - string
    Output:
        beat_vec - list[int] of 3 elements
    '''
    #print(self.stream.parts[0][0].timeSignature.ratioString)
    #print(beat_num)
    try:
        beat_pos = int(beat_pos)
    except ValueError:
        beat_pos = float(beat_pos)

    strong, weak = define_beat_from_meter(meter)

    if beat_pos in strong:
        beat_vec = [1,0,0]
    elif beat_pos in weak:
        beat_vec = [0,1,0]
    else:
        beat_vec = [0,0,1]
    #print(beat_vec)
    return beat_vec 


def get_onset_notes(cur_onset_notes, prev_onset_notes):
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


def to_12d_vec(note_list):
    '''
    one-hot encoding on 12 dimension (12 semitones) with input of a list in note name

    Input: 
        note_list - list[string]
    Output:
        output_vec - list[int] of 12 elements
    '''
    output_vec = [0]*12
    for midi in note_list:
        n = midi%12
        if output_vec[n] == 0:
            output_vec[n] = 1
    return output_vec

def process_note(note):
    '''
    Process note name

    Input: 
        note - string
    Output:
        out_note - string
    '''
    d = {'C--':'B-', 'C##': 'D#', 'D--':'C', 'D##':'E', 'E--':'D', 'E##':'F#', 'F--':'E-', 'F##':'G', 
    'G--':'F', 'G##':'A', 'A--':'G', 'A##':'B', 'B--':'A', 'B##':'C#'}
    if '-' in note:
        cnt = len(re.findall("-", note))
        if cnt == 1:
            out_note = "".join(dict.fromkeys(note))
        else:
            l = re.sub('[^a-gA-G]+', '', note)
            letter = "".join(dict.fromkeys(l))
            tmp_note = letter + cnt * '-'
            tmp_note = tmp_note.upper()
            out_note = d[tmp_note]

    elif '#' in note:
        cnt = len(re.findall("#", note))
        if cnt == 1:  
            out_note = "".join(dict.fromkeys(note))
        else:
            l = re.sub('[^a-gA-G]+', '', note)
            letter = "".join(dict.fromkeys(l))
            tmp_note = letter + cnt * '#'
            tmp_note = tmp_note.upper()
            out_note = d[tmp_note]
    else:
        out_note = "".join(dict.fromkeys(note))
    out_note = out_note.upper()
    return out_note


def to_21d_vec(note_list):
    '''
    one-hot encoding on 21 dimension with input of a list in note name

    Input: 
        note_list - list[string]
    Output:
        output_vec - list[int] of 21 elements
    '''
    dic = {'C':0, 'C#':1, 'D-':2, 'D':3, 'D#':4, 'E-':5, 'E':6, 'E#':7, 'F-':8, 'F':9, 'F#':10, 
    'G-':11, 'G':12, 'G#':13, 'A-':14, 'A':15, 'A#':16, 'B-':17, 'B':18, 'B#':19, 'C-':20}

    out_vec = [0]*21

    for note in note_list:
        note_name = process_note(note)
        idx = dic[note_name]
        if out_vec[idx] == 0:
            out_vec[idx] = 1
    return out_vec

def specialChords(chord):
    if chord == "C64" or chord == 'C':
        output = 'Cad'
    elif chord == "Gr":
        output = "Gn"
    else:
        output = chord
    return output

#print(to_21d_vec(['A', 'e', 'c#', 'a']))
        