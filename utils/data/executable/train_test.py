# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

import torch
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # n = 10
    # features = 2
    # X, y = torch.arange(n * features).reshape(n, features)+1, torch.tensor([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    #
    # print(X)
    # # array([[0, 1],
    # #        [2, 3],
    # #        [4, 5],
    # #        [6, 7],
    # #        [8, 9]])
    #
    # print(list(y))
    # # [0, 1, 2, 3, 4]
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    #
    # X_train = X_train * 2
    #
    # print("X_train:", X_train)
    # print("X:", X_train)
    #
    # print("y_train:", y_train)
    #
    # print("X_test:", X_test)
    #
    # print("y_test:", y_test)

    indices = list(range(10))
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )

    print(train_idx)
    print(test_idx)
