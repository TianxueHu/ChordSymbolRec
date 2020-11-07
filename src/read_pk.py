import pickle
objects = []
with (open("src/pickle_vector_lists/bach_org_vectors_s2s_21enc_4meaWindow.pkl", "rb")) as openfile:
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
