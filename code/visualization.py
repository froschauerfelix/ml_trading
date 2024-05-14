import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_trading.code.config import full_path


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification
import pandas as pd
np.random.seed(41)


support = True
if support:

    # Generate random data
    num_points = 10
    x_1 = np.random.normal(1, 1, num_points)
    x_2 = np.random.normal(1, 2, num_points)

    x_3 = np.random.normal(2.5, 0.5, num_points)
    x_4 = np.random.normal(2, 1, num_points)

    data = pd.concat([
        pd.DataFrame({'x_1': x_1, 'x_2': x_2, 'y': 0}),
        pd.DataFrame({'x_1': x_3, 'x_2': x_4, 'y': 1})
    ])

    X = data[['x_1', 'x_2']]
    y = data['y']

    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)

    # Get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(X['x_1'].min(), X['x_1'].max())
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # Calculate margin lines
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy + np.sqrt(1 + a ** 2) * margin
    yy_up = yy - np.sqrt(1 + a ** 2) * margin

    # Plot the data points and the separating hyperplane
    plt.figure(figsize=(10, 8))

    # Plot data points
    plt.scatter(X['x_1'][y == 0], X['x_2'][y == 0], color='red', label='Class 0', marker="x")
    plt.scatter(X['x_1'][y == 1], X['x_2'][y == 1], color='blue', label='Class 1', marker="x")

    # Plot the separating hyperplane
    plt.plot(xx, yy, 'k-', label='Separating hyperplane')

    # Plot margin lines
    plt.plot(xx, yy_down, 'k--', label='Margin')
    plt.plot(xx, yy_up, 'k--')

    # Highlight the support vectors
    #plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
             #   s=100, facecolors='none', edgecolors='k', label='Support vectors')

    #plt.title('SVM Decision Boundary with Support Vectors and Margins')
    plt.xlabel('Input Feature 1')
    plt.ylabel('Input Feature 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(full_path + "output/svm_visualization.svg")
    plt.show()



