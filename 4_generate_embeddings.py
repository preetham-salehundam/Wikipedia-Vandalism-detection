"""
@author: Preetham Salehundam
@email: salehundam.2@wright.edu

script to generate node2vec embeddings
"""
from node2vec import Node2Vec
import json
import networkx as nx
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from stellargraph import StellarGraph
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


if __name__ == "__main__":
    data = json.loads(open("graph.json").read())
    graph = nx.node_link_graph(data)
    G = StellarGraph.from_networkx(graph)
    print(G.info())
    n2v = Node2Vec(graph, dimensions=128, workers=4, num_walks=10, walk_length=10)
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    print(model)
    ordered_vocab = [(term, voc.index, voc.count) for term, voc in model.wv.vocab.items()]
    ordered_vocab = sorted(ordered_vocab, key=lambda k: k[2])
    ordered_terms, term_indices, term_counts = zip(*ordered_vocab)
    word_vectors = pd.DataFrame(model.wv.syn0[term_indices, :], index=ordered_terms)
    word_vectors.to_csv("data_embds.csv", index=False)
    #labels = word_vectors
    pca = PCA(n_components=2)
    components = pca.fit_transform(word_vectors)
    _class= [graph.nodes[int(node_id)]["_class"] for node_id in ordered_terms]
    components_2d = pd.DataFrame({1:components[:,0], 2:components[:,1], "class":_class}, index=ordered_terms)
    components_2d.plot.scatter(x=1, y=2, c="class",colormap ="Set1")
    components = pd.DataFrame(components, index=ordered_terms)
    plt.legend()
    plt.show()
    trans = TSNE(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(word_vectors), index=ordered_terms)
    emb_transformed["label"] = _class
    alpha = 0.7

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        emb_transformed[0],
        emb_transformed[1],
        c=emb_transformed["label"].astype("category"),
        cmap="jet",
        alpha=alpha,
    )
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    plt.title(
        "{} visualization of GraphSAGE embeddings for cora dataset".format(TSNE.__name__)
    )
    plt.legend()
    plt.show()

    raw_data = pd.read_csv("category_data_new.csv")
    raw_data = raw_data.set_index("editid")
    # word vector index is str
    word_vectors.index = word_vectors.index.astype("int64")
    raw_data = raw_data.join(word_vectors, how="inner")
    y = raw_data["class"]
    data = raw_data.drop(["class", "oldrevisionid", "editor", "newrevisionid", "category", "articletitle", "articleid"], axis=1)
    data = PCA().fit_transform(data)
    data_x = StandardScaler().fit(data).transform(data)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf.get_n_splits(data_x, y)

    X_train, X_test = train_test_split(np.array(data_x), test_size=0.2, random_state=123)
    Y_train, Y_test = train_test_split(np.array(y), test_size=0.2, random_state=123)

    rf = RandomForestClassifier(class_weight="balanced", max_depth=5, n_estimators=1)
    lr = LogisticRegression(class_weight="balanced", penalty="l2")
    print(cross_val_score(rf, data_x, y, scoring="f1_macro", cv=5))
    print(cross_val_score(lr, data_x, y, scoring="f1_macro", cv=5))
    i = 0
    for train_index, test_index in skf.split(data_x, y):
        print("iteration ", i)
        i += 1
        X_train, X_test = data_x[train_index], data_x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rf.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        importances = {col: imp for col, imp in zip(data.columns, rf.feature_importances_)}
        print(sorted(importances.items(), key=lambda x: -x[1]))
        importances = {col: imp for col, imp in zip(data.columns, lr.coef_[0])}
        print(sorted(importances.items(), key=lambda x: -x[1]))
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        print("Randomforest-train,\n", classification_report(y_true=y_train, y_pred=y_pred_train))
        print("Randomforest-test,\n", classification_report(y_true=y_test, y_pred=y_pred_test))
        y_pred_train = lr.predict(X_train)
        y_pred_test = lr.predict(X_test)
        print("Logistic Regression-train,\n", classification_report(y_true=y_train, y_pred=y_pred_train))
        print("Logistic Regression-test,\n", classification_report(y_true=y_test, y_pred=y_pred_test))

    # HARD LUCK WITH TSNE
    # visualization
    # trans  = TSNE(n_components=2, perplexity=10, n_iter_without_progress=10)
    # emb_trans = pd.DataFrame(trans.fit_transform(word_vectors))
    #
    #
    # alpha = 0.7
    #
    # fig, ax = plt.subplots(figsize=(14, 8,))
    # ax.scatter(emb_trans[0], emb_trans[1] , c= np.hstack((train_labels.reshape(-1,) , test_labels.reshape(-1,))), cmap="jet", alpha=alpha)
    # ax.set(xlabel="$X_1$", ylabel="$X_2$")
    # plt.title('{} visualization of embeddings for wiki dataset'.format(TSNE.__name__), fontsize=24)
    # plt.show()

    print("embedding generation complete!!")