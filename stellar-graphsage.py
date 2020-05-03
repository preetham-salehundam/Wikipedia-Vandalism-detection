"""
@author: Preetham Salehundam
@email: salehundam.2@wright.edu
script to apply graphsage on the graph.json data
"""

from stellargraph import *
import networkx as nx
import json
import numpy as np
from stellargraph.mapper import GraphSAGENodeGenerator, HinSAGELinkGenerator
from stellargraph.layer import GraphSAGE
from random import shuffle
from tensorflow.keras import layers, Model, optimizers, losses, callbacks
from stellargraph.data import BiasedRandomWalk, UniformRandomWalk
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn import model_selection

from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from node2vec import Node2Vec


# class EarlyStop(callbacks.CallBack):
#     def on_epoch_end(self, batch, logs=None):
#         print(logs["loss"], "test")




if __name__ == "__main__":

    # do something
    data = json.loads(open("data/graph_cat.json").read())
    graph = nx.node_link_graph(data)
    # print(graph)
    # data = {}
    # for node in graph.nodes:
    #
    #     data[node] = np.zeros(shape=100,)
    #graph = graph.to_undirected()
    G = StellarGraph.from_networkx(graph, node_features="feature")
    print(G.node_types)
    G.check_graph_for_ml()
    nodes = [node for node in graph.nodes]
    shuffle(nodes)
    train_ids = nodes[:5000]
    test_ids = nodes[5000:]
    train_labels= [graph.nodes[id]["_class"] for id in train_ids]
    test_labels = [graph.nodes[id]["_class"] for id in test_ids]
    all_labels = train_labels + test_labels
    train_labels = np.array(train_labels).reshape(len(train_ids),1)
    test_labels = np.array(test_labels).reshape(len(test_ids), 1)
    print(np.unique(train_labels, return_counts=True))
    print(np.unique(test_labels, return_counts=True))
    generator = GraphSAGENodeGenerator(G, batch_size=50, num_samples=[10,10])
    train_data_gen = generator.flow(train_ids, train_labels)
    test_data_gen = generator.flow(test_ids, test_labels)
    all_gen = generator.flow(list(nodes), all_labels)

    print("Node Gen done!")
    base_model = GraphSAGE(layer_sizes=[32, 32], generator=generator, bias=True, dropout=0.8)
    x_in, x_out = base_model.build()
    prediction = layers.Dense(units=2, activation="softmax")(x_out)

    print("model building done")

    model = Model(inputs=x_in, outputs = prediction)
    model.compile(optimizer=optimizers.Adam(lr=0.005), loss=losses.categorical_crossentropy, metrics=["acc"])
    tensorboard = callbacks.TensorBoard(log_dir="logs",embeddings_freq=1, update_freq=1, histogram_freq=1)
    tboard = model.fit(train_data_gen, epochs=4, validation_data=test_data_gen, verbose=True,
                                  shuffle=False, callbacks=[tensorboard])
    print(tboard)
    print("prediction done")

    y_pred = model.predict(train_data_gen, verbose=1)
    labels = np.argmax(y_pred, axis=1)
    print(classification_report(labels, train_labels))


    y_pred = model.predict(test_data_gen, verbose=1)
    labels = np.argmax(y_pred, axis=1)
    print(classification_report(labels, test_labels))

    print(model.layers)
    emb_model = Model(inputs=x_in, outputs=model.layers[-4].output)
    embs=emb_model.predict_generator(generator=all_gen)
    #
    # node_subject = all_labels
    # all_node_ids = train_ids + train_ids
    # all_node_ids =  np.array(all_node_ids).reshape(-1,1)

    # X = embs.reshape(-1,32)
    # if X.shape[1] > 2:
    #     transform = TSNE  # PCA
    #
    #     trans = transform(n_components=2)
    #     emb_transformed = pd.DataFrame(trans.fit_transform(X), index=all_node_ids)
    #     emb_transformed["label"] = node_subject
    # else:
    #     emb_transformed = pd.DataFrame(X, index=all_node_ids)
    #     emb_transformed = emb_transformed.rename(columns={"0": 0, "1": 1})
    #     emb_transformed["label"] = node_subject
    #
    # alpha = 0.7
    #
    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.scatter(
    #     emb_transformed[0],
    #     emb_transformed[1],
    #     c=emb_transformed["label"].astype("category"),
    #     cmap="jet",
    #     alpha=alpha,
    # )
    # ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    # plt.title(
    #     "{} visualization of GraphSAGE embeddings for cora dataset".format(transform.__name__)
    # )
    # plt.show()

    train_embs = embs[:5000]
    test_embs = embs[5000:]
    train_embs = train_embs.reshape(-1,32)
    test_embs = test_embs.reshape(-1, 32)
    #class_weight= compute_class_weight ("balanced", np.unique(train_labels), train_labels)
    lr = LogisticRegressionCV(cv=5, class_weight="balanced", max_iter=10000)
    lr.fit(train_embs, train_labels)
    train_probs = lr.predict(train_embs)
    print(lr.score(train_embs, train_probs))
    test_probs = lr.predict(test_embs)
    y_pred = np.where(test_probs > 0.5, 1, 0)
    print(lr.score(test_embs, test_probs))


    #model.predict_generator(generator=generator)



    # node2vec

    # gen random walks
    # biased_walk  = BiasedRandomWalk(G)
    # #nodes = [str(id) for id in train_ids + test_ids]
    # nodes= train_ids + test_ids
    # print(nodes)
    # walks=biased_walk.run(nodes= nodes ,p=0.5, n=3, length=10)
    # model = Word2Vec(sentences=walks, size=128, window=10, min_count=0, sg=1, iter=1)
    # #model.wv.save_word2vec_format("data.emb")
    # ordered_vocab = [(term, voc.index, voc.count) for term, voc in model.wv.vocab.items()]
    # ordered_vocab = sorted(ordered_vocab, key=lambda k: k[2])
    # ordered_terms, term_indices, term_counts = zip(*ordered_vocab)
    # word_vectors = pd.DataFrame(model.wv.syn0[term_indices, :], index=ordered_terms)
    #
    # # visualization
    # trans  = TSNE(n_components=2, perplexity=10, n_iter_without_progress=10)
    # emb_trans = pd.DataFrame(trans.fit_transform(word_vectors))
    #
    #
    # alpha = 0.7
    #
    # fig, ax = plt.subplots(figsize=(14, 8,))
    # ax.scatter(emb_trans[0], emb_trans[1] , c= np.hstack((train_labels.reshape(-1,) , test_labels.reshape(-1,))), cmap="jet", alpha=alpha)
    # ax.set(xlabel="$X_1$", ylabel="$X_2$")
    # plt.title('{} visualization of embeddings for tweeter dataset'.format(TSNE.__name__), fontsize=24)
    # plt.show()
    # pass

    # n2v = Node2Vec(graph, dimensions=20, walk_length=30, num_walks=50, workers=4)
    # model = n2v.fit(window=10, min_count=1, batch_words=4)
    #
    # model = RandomForestClassifier().fit()
    # model.predict()
    # print(confusion_matrix(y_pred, test_labels))
