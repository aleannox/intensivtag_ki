from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_tsne(labels, embedding_vectors, perplexity=40):
    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca',
                      n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(embedding_vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(12, 12))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], s=0.8)
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontsize=8)
    plt.show()
