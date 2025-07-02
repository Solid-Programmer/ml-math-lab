import os
import pickle

def load_model(path="model/naive_bayes_model.pkl"):
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'naive_bayes_model.pkl'))
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
