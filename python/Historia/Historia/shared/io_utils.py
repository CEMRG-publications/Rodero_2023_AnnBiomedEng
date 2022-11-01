import pickle


def save(obj, filename):
    with open(filename + ".pickle", "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return


def load(filename):
    with open(filename + ".pickle", "rb") as f:
        obj = pickle.load(f)
    return obj
