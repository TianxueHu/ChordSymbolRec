# Project log
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