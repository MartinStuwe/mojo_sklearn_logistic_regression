
from python import Python

fn main() raises:
    let sklearn_datasets = Python.import_module("sklearn.datasets")
    let iris = sklearn_datasets.load_iris()

    let X = iris['data']
    let Y = iris['target']

    let n_samples = X.shape[0]
    let n_features = X.shape[1]

    print("Amount of samples: ", n_samples)
    print("Amount of features: ", n_features)

    let sklearn_linear_model = Python.import_module("sklearn.linear_model")
    let clf = sklearn_linear_model.LogisticRegression()   

    let sklearn_model_selection = Python.import_module("sklearn.model_selection")
    let scores = sklearn_model_selection.cross_val_score(clf, X, Y)

    print("Accuracy:", scores.mean())
