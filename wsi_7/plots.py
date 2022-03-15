import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import confusion_matrix

def scatter_plot(data):
    sepal_lengths = [d[0] for d in data]
    sepal_widths = [d[1] for d in data]
    petal_lengths = [d[2] for d in data]
    petal_widths = [d[3] for d in data]
    target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    targets=[]
    for i in range(0, 3):
        for _ in range(0, 50):
            targets.append(i)

    # this formatter will label the colorbar with the correct target names
    formatter = plt.FuncFormatter(lambda i, *args: target_names[int(i)])

    plt.figure(figsize=(5, 4))
    plt.scatter(sepal_lengths, sepal_widths, c=targets)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel("sepal length [cm]")
    plt.ylabel("sepal width [cm]")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 4))
    plt.scatter(petal_lengths, petal_widths, c=targets)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel("petal length [cm]")
    plt.ylabel("petal width [cm]")
    plt.tight_layout()
    plt.show()

def confusion_matrix_plot(actual, predicted, labels):
    mat = confusion_matrix(actual, predicted, labels=labels)
    s = sn.heatmap(mat, annot=True)
    s.set_title("Confusion matrix")
    s.set_xticklabels(labels)
    s.set_yticklabels(labels)
    plt.show()