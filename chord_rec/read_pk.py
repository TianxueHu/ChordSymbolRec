import pickle
import h5py
'''
objects = []
with (open("src/pickle_vector_lists/Sears_reduce_vectors_ffnn_21enc.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

for piece in objects:
    #print("pieces:",len(piece))
    print(piece)
    for windows in piece:
        #print("Number of onsets:",len(num_onsets))
        print(windows)
        #for l in num_onsets:
        #    print("len of vectors in each onset: ", l)
'''
with h5py.File('Bach_org_vectors_ffnn_21enc.h5', 'r') as hf:
    data = hf
print(data)