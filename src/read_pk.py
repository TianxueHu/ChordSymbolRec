import pickle
objects = []
with (open("src/pickle_vector_lists/Sears_reduce_vectors_ffnn_21enc.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

for piece in objects:
    #print("pieces:",len(piece))
    for num_onsets in piece:
        #print("Number of onsets:",len(num_onsets))
        
        for l in num_onsets:
            print("len of vectors in each onset: ", l)
