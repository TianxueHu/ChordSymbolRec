import re
from music21 import *

quality_transfer = {
    'half-diminished seventh' : 'm7b5', 
    'minor' : 'm', 
    'minor seventh' : 'm7', 
    'fully-diminished seventh' : 'dim7', 
    'major seventh' : 'M7', 
    'french augmented sixth' : 'Fr+6', 
    'diminished' : 'dim', 
    'italian augmented sixth' : 'It+6', 
    'augmented' : 'aug', 
    'minor major seventh' : 'mM7', 
    'dominant seventh' : '7',
    'augmented major seventh' : '+M7', 
    'german augmented sixth' : 'Gr+6', 
    'major' : ''
}


def convert_chord(chord):
    splitted = chord.split(' ')
    root = str(splitted[0])
    quality = ' '. join(splitted[1:])
    
    if 'x' in root:
        root = re.sub(r'x', '##', root)
    if 'b' in root:
        root = re.sub(r'b', '-', root)
    
    new_qual = quality_transfer[quality]    
    new_chord = root + new_qual    
    return new_chord


def chord_similarity(label, pred):
    label = convert_chord(label)
    pred = convert_chord(pred)
    label_notes = None
    pred_notes = None
    
    if label == 'GGr+6': # handle bug in music21
        label_notes = harmony.ChordSymbol(root = 'G', kind = 'German')
    if pred == 'GGr+6':
        pred_notes = harmony.ChordSymbol(root = 'G', kind = 'German')
    if not label_notes:
        label_notes = harmony.ChordSymbol(label)
    if not pred_notes:
        pred_notes = harmony.ChordSymbol(pred)
        
    label_notes = [str(n.name) for n in label_notes.notes]
    pred_notes = [str(n.name) for n in pred_notes.notes]
    # print(label_notes, pred_notes)
    inter = set(label_notes).intersection(pred_notes) 
    union = set(label_notes).union(pred_notes) 
    distance = len(inter)/len(union)
    return distance


#print(chord_distance('Bbb major', 'B minor seventh'))