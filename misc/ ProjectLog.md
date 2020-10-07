# Project log
## 10/7
- [TODO] Process vectors for FFNN
- [TODO] Orgranize and process availiable pieces that Craig Sapp in the ABC dataset
- [TODO] Orgranize and process David Sears's dataset

So far the datasets we have: <br>
- Haydn op20
- Bach Chorale 69 pieces
- [TODO] ABC 
- [TOOD] David Sear's dataset

Some experiments plans: <br>
 - Compare the performances with/without NCT eleminated
 - Compare the performances using 12 entries (with post-processing) input and 21 entries input.


## 10/2
- Augmented Bach chorals (69 songs) and Haydn sun quartets.
Located at
`/dataset/haydn_op20_harm/haydn_final_scores` and `/datasets/bhchorale/bach_final_scores`
- Aligning the new Bach chorales dataset (140 songs). 
- Parsed ABC dataset from .mscx to .musicxml, failed to convert to .krn since musicxml2krn command doesn't work for harmony. [PENDING]
## 9/22
Pre-process Bach Chorale Dataset - removed NCT and generate new scores
## 9/12
Pre-process Haydn Dataset:
- Take out each part as melody to feed in NCT model (Humdrum).
- Modify NCT script in R for this project.
- Feed each melody into the model and generate reduced scores, if a note is identified as NCT, change it to a rest
    - For two or more notes played at the same time in string quartets, we take the bass note to calculate the intervals.
    - For two or more notes played at the same time in string quartets, if this "chordal" notes are not identified as NCT, keep the original chordal note in the new score, otherwise, change them to a rest.

New scores are located in /dataset/haydn_op20_harm/haydn_reduced_scores

## Until  9/7
Define what to do for the project 
- Improvements upon previous work
- Project Proposal
- Dataset