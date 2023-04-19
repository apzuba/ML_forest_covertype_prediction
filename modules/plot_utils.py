
import matplotlib.pyplot as plt


def plot_accuracies(spearm_eval, ovr_eval, svm_eval, tf_eval):

    # accuracy scores
    scores = [spearm_eval, ovr_eval, svm_eval, tf_eval]

    # models
    models = ['Spearman', 'OvR', 'SVM', 'Tensorflow']

    # plot
    fig, ax = plt.subplots()
    ax.bar(models, scores)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy Score')
    ax.set_title('Comparison of Model Accuracies')

    # add scores with values to the second degree on top of the bars
    for i, v in enumerate(scores):
        ax.text(i, v+0.01, str(round(v, 2)), ha='center')
        
    plt.show()







def plot_k_means(km, X, y_km):
    # plot the 3 clusters
    plt.scatter(
        X[y_km == 0, 0], X[y_km == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='cluster 1'
    )

    plt.scatter(
        X[y_km == 1, 0], X[y_km == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )

    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=50, c='lightblue',
        marker='v', edgecolor='black',
        label='cluster 3'
    )
    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=50, c='black',
        marker='v', edgecolor='black',
        label='cluster 4'
    )
    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=50, c='purple',
        marker='v', edgecolor='black',
        label='cluster 5'
    )
    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=50, c='yellow',
        marker='v', edgecolor='black',
        label='cluster 6'
    )
    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=50, c='red',
        marker='v', edgecolor='black',
        label='cluster 7'
    )

    # plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
    )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()