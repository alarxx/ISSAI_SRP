# KFold:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# StratifiedKFold:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

if __name__ == "__main__":
    X = np.array([[10, 20], [30, 40], [11, 21], [31, 41], [51, 61], [71, 81]])
    y = np.array([0, 0, 1, 1, 1, 1])

    # skf = KFold(n_splits=2)
    skf = StratifiedKFold(n_splits=2)
    print(skf.get_n_splits(X, y))

    print(skf)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}, labels: {y[train_index]}")
        print(f"  Test:  index={test_index}, labels: {y[train_index]}")
    # Fold 0:
    #   Train: index=[1 3]
    #   Test:  index=[0 2]
    # Fold 1:
    #   Train: index=[0 2]
    #   Test:  index=[1 3]
