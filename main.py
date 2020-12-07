import networkx
import pandas as pd
import numpy as np
import random
import networkx as nx
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.utils import class_weight

from node2vec import Node2Vec
import tensorflow as tf
from functools import partial


def main(which):

    df1, df2, data, x_test, y_test = write_to_file(which)

    file_loc = "Data/" + which + "/saves/"
    with open(file_loc + "xtrain.txt") as file:
        xtrain = [[float(digit) for digit in line.split()] for line in file]
    with open(file_loc + "xtest.txt") as file:
        xtest = [[float(digit) for digit in line.split()] for line in file]
    with open(file_loc + "ytrain.txt") as file:
        ytrain = [[float(digit) for digit in line.split()] for line in file]
    with open(file_loc + "ytest.txt") as file:
        ytest = [[float(digit) for digit in line.split()] for line in file]

    logreg_predictions = logreg(xtrain, xtest, ytrain, ytest)
    make_graphs(logreg_predictions, x_test, ytest, data)

    knn_predictions = knn(xtrain, xtest, ytrain, ytest)
    make_graphs(knn_predictions, x_test, ytest, data)


def make_graphs(predictions, x_test, ytest, data):
    t_p1 = []
    t_p2 = []
    f_p1 = []
    f_p2 = []
    f_n1 = []
    f_n2 = []

    node_numbers = x_test.index

    j = 0
    for i in range(len(node_numbers)):
        node_number = node_numbers[i]
        row = data.loc[node_number]
        node1 = row['node_1']
        node2 = row['node_2']
        prediction = predictions[i]
        link = int(ytest[i][0])
        if prediction == 1 and link == 1:
            t_p1.append(node1)
            t_p2.append(node2)
        elif prediction == 0 and link == 1:
            f_n1.append(node1)
            f_n2.append(node2)
        elif prediction == 1 and link == 0:
            f_p1.append(node1)
            f_p2.append(node2)
        else:
            j += 1
    true_p = pd.DataFrame({'node_1': t_p1,
                           'node_2': t_p2})

    false_p = pd.DataFrame({'node_1': f_p1,
                            'node_2': f_p2})

    false_n = pd.DataFrame({'node_1': f_n1,
                            'node_2': f_n2})

    true_p['color'] = 'black'
    false_n['color'] = 'r'
    false_p['color'] = 'c'

    frames = [true_p, false_n]
    GG = pd.concat(frames, ignore_index=True)

    Graph = nx.from_pandas_edgelist(GG, "node_1", "node_2", "color", create_using=nx.Graph())

    edges = Graph.edges()
    colors = [Graph[u][v]['color'] for u, v in edges]

    plt.figure(figsize=(10, 10))

    pos = nx.random_layout(Graph, seed=23)
    nx.draw(Graph, with_labels=False, pos=pos, node_size=40, alpha=0.6, width=0.7, edge_color=colors)
    plt.show()


def write_to_file(which):
    node_list_1 = []
    node_list_2 = []

    if which == 'facebook':
        with open("Data/facebook/fb-pages-food.nodes", encoding="utf-8") as f:
            nodes = f.read().splitlines()
        with open("Data/facebook/fb-pages-food.edges") as f:
            links = f.read().splitlines()

        for i in tqdm(links):
            node_list_1.append(i.split(',')[0])
            node_list_2.append(i.split(',')[1])

    elif which == 'biogrid':

        with open("Data/Biogrid/Biogrid.edgelist") as f:
            links = f.read().splitlines()

        for i in tqdm(links):
            node_list_1.append(i.split()[0])
            node_list_2.append(i.split()[1])

    df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})

    # create graph
    G = nx.from_pandas_edgelist(df, "node_1", "node_2", create_using=nx.Graph())

    # plot graph
    plt.figure(figsize=(10, 10))

    pos = nx.random_layout(G, seed=23)
    nx.draw(G, with_labels=False, pos=pos, node_size=40, alpha=0.6, width=0.7)

    plt.show()

    # combine all nodes in a list
    node_list = node_list_1 + node_list_2

    # remove duplicate items from the list
    node_list = list(dict.fromkeys(node_list))

    # build adjacency matrix
    adj_G = nx.to_numpy_matrix(G, nodelist=node_list)

    # get unconnected node-pairs
    all_unconnected_pairs = []

    # traverse adjacency matrix
    offset = 1
    for i in tqdm(range(1, adj_G.shape[0])):
        for j in range(offset, adj_G.shape[1]):
            if i != j:
                try:
                    path_length = nx.shortest_path_length(G, str(i), str(j))
                except networkx.exception.NetworkXNoPath:
                    path_length = df.size
                # if nx.shortest_path_length(G, str(i), str(j)) <= 2:
                #     if adj_G[i, j] == 0:
                #         all_unconnected_pairs.append([node_list[i], node_list[j]])
                if path_length <= 2:
                    if adj_G[i, j] == 0:
                        all_unconnected_pairs.append([node_list[i], node_list[j]])

        offset = offset + 1

    # print(len(all_unconnected_pairs))

    node_1_unlinked = [i[0] for i in all_unconnected_pairs]
    node_2_unlinked = [i[1] for i in all_unconnected_pairs]

    data = pd.DataFrame({'node_1': node_1_unlinked,
                         'node_2': node_2_unlinked})

    # add target variable 'link'
    data['link'] = 0

    initial_node_count = len(G.nodes)

    fb_df_temp = df.copy()

    # empty list to store removable links
    omissible_links_index = []

    for i in tqdm(df.index.values):

        # remove a node pair and build a new graph
        G_temp = nx.from_pandas_edgelist(fb_df_temp.drop(index=i), "node_1", "node_2", create_using=nx.Graph())

        # check there is no spliting of graph and number of nodes is same
        if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
            omissible_links_index.append(i)
            fb_df_temp = fb_df_temp.drop(index=i)

    # create dataframe of removable edges
    fb_df_ghost = df.loc[omissible_links_index]

    # add the target variable 'link'
    fb_df_ghost['link'] = 1

    data = data.append(fb_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)

    df_copy = df.copy()
    # drop removable edges
    fb_df_partial = df_copy.drop(index=fb_df_ghost.index.values)

    # build graph
    G_data = nx.from_pandas_edgelist(fb_df_partial, "node_1", "node_2", create_using=nx.Graph())

    # plot graph
    plt.figure(figsize=(10, 10))

    pos = nx.random_layout(G, seed=23)
    nx.draw(G_data, with_labels=False, pos=pos, node_size=40, alpha=0.6, width=0.7)

    plt.show()


    # Generate walks
    node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50)

    # train node2vec model
    n2w_model = node2vec.fit(window=7, min_count=1)

    x = [(n2w_model[str(i)] + n2w_model[str(j)]) for i, j in zip(data['node_1'], data['node_2'])]

    # sc = StandardScaler()
    # sc.fit_transform(x)
    X = pd.DataFrame(x)

    # print(X)
    xtrain, xtest, ytrain, ytest = train_test_split(X, data['link'], test_size=0.3, random_state=35)
    # print(xtest)
    file_loc = "Data/" + which + "/saves/"
    np.savetxt(file_loc + "xtrain.txt", xtrain)
    np.savetxt(file_loc + "xtest.txt", xtest)
    np.savetxt(file_loc + "ytrain.txt", ytrain)
    np.savetxt(file_loc + "ytest.txt", ytest)

    return df, fb_df_partial, data, xtest, ytest


def logreg(xtrain, xtest, ytrain, ytest):
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)

    lr.fit(xtrain, ytrain)

    predictions = lr.predict(xtest)

    cm = confusion_matrix(ytest, predictions, labels=[1, 0])
    # print(roc_auc_score(ytest, predictions[:, 1]))
    print(accuracy_score(ytest, predictions))
    # print(balanced_accuracy_score(ytest, predictions))
    print(cm)
    return predictions


def knn(xtrain, xtest, ytrain, ytest):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(input_shape=(100,), units=1000, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        tf.keras.layers.Dense(units=100, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=100, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer='adam',
                  # loss=tf.reduce_all_mean(tf),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC()])
                  # metrics=[tf.keras.metrics.Accuracy()])

    # ytrain = np.array(ytrain).ravel()
    # # xtrain = np.array(xtrain).ravel()
    # # xtest = np.array(xtest).ravel()
    # ytest = np.array(ytest).ravel()
    # print(ytrain)
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(np.array(ytrain).ravel()), np.array(ytrain).ravel())
    # class_weights = dict(enumerate(class_weights))

    model.fit(xtrain, ytrain, epochs=50, batch_size=1)
    predictions = model.predict(xtest)
    model.evaluate(xtest, ytest)
    predictions = [1 if predictions[i] > 0.5 else 0 for i in range(len(predictions))]
    print(tf.math.confusion_matrix(ytest, predictions))
    return predictions
    # text(str(body[0]) + " " + str(body[1]))


def text(body):
    import smtplib
    email = "pythonsmsjulian@gmail.com"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(email, "JulianPythonSMS")
    # body = "All Done\n"
    server.sendmail(email, '7733267749@vtext.com', body)
    server.quit()


def pretty_print(matrix):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(pd.DataFrame(matrix))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('facebook')
    # main('biogrid')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
