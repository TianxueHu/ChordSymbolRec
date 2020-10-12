import pickle
objects = []
with (open("Haydn_vectors_ffnn_21enc.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
for o in objects:
    print(len(o))