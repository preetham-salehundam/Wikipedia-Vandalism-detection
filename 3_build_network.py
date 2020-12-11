"""
@author: Preetham Salehundam
@email: salehundam.2@wright.edu
This script will build a networkx graph from the data and serializes it to graph.json
this also writes the graph to neo4j database
"""
import pandas as pd
import networkx as nx
import json
from networkx.readwrite import json_graph
import shutil
from tqdm import tqdm
from stellargraph import StellarGraph
import numpy as np
import py2neo
from py2neo import Node, Relationship
from itertools import combinations
import random



def get_user_article_relation(data_file):
    """
    :param data_file:
    :return:
    """
    raw_data = pd.read_csv(data_file)
    if "class" not in raw_data.columns:
        raw_data['class'] = raw_data.loc[raw_data["vulgar_word_ratio"] != 0] = "vandalism"
        raw_data['class'] = raw_data.loc[raw_data["vulgar_word_ratio"] == 0] = "regular"

    raw_data.loc[raw_data["class"] == "vandalism", "class"] = 1
    raw_data.loc[raw_data["class"] == "regular", "class"] = 0
    return raw_data


def create_network(data):
    """
    :param data:
    :return:
    """
    graph = nx.Graph()
    for _, row in tqdm(data.iterrows(), ascii=True, desc="processing rows"):
        graph.add_node(row.editid, label="editor", feature=[row.character_diversity, row.digit_ratio,
                                                             row.non_alpha_numeric_ratio, row.upper_to_all,
                                                             row.upper_to_lower, row.size_increment, row.comment_len,
                                                             row.vulgar_word_ratio, row.longest_Word, row.bad_words])
        graph.add_node(row.articleid, label="article", feature=[row.size, row.word_count])
        graph.add_edge(row.editid, row.articleid, label="edit", feature=[row["class"]])
    return graph


def shortcircuit(data):
    """
    :param data:
    :return:
    """
    graph = nx.Graph()
    for _, row in tqdm(data.iterrows(), ascii=True, desc="processing rows"):
        graph.add_node(row.editid, label="editor", feature=[row.character_diversity, row.digit_ratio,
                                                            row.non_alpha_numeric_ratio, row.upper_to_all,
                                                            row.upper_to_lower, row.size_increment, row.comment_len,
                                                            row.vulgar_word_ratio, row.longest_Word, row.bad_words],
                       group="editor", _class=row["class"])
    if "category" in row and row["category"] is not np.nan:
        for key, group_df in data.groupby('category')["editid"]:
            nodes = group_df.to_list()
            if len(nodes) > 2:
                comb = random.sample(nodes, 2)
                graph.add_edge(comb[0], comb[1], group='category')

    for key, group_df in data.groupby('articleid')["editid"]:
        for comb in combinations(group_df.to_list(),2):
            graph.add_edge(comb[0], comb[1], group="article")

    return graph

def create_dummy_network(data):
    """
    :param data:
    :return:
    """
    print("creating network with 0 feature vec")
    graph = nx.Graph()
    BELONGS_TO = Relationship.type("BELONGS_TO")
    EDITED_BY = Relationship.type("EDITED_BY")
    category_edge = None
    editor_article_edge = None

    # driver = GraphDatabase("bolt://localhost:7687", auth=("wikipedia", "1234"), encrypted=True)
    tx = py2neo.Graph("bolt://wikipedia:1234@localhost:7687").begin()
    for _, row in tqdm(data.iterrows(), ascii=True, desc="processing rows"):
        editor = Node("Editor", name=str(row.editid), node_id=row.editid, label=row.editor, feature=[row.character_diversity, row.digit_ratio,
                                                             row.non_alpha_numeric_ratio, row.upper_to_all,
                                                             row.upper_to_lower, row.size_increment, row.comment_len,
                                                             row.vulgar_word_ratio, row.longest_Word, row.bad_words], group="editor", _class=row["class"])
        graph.add_node(row.editid, label="editor", feature=[row.character_diversity, row.digit_ratio,
                                                             row.non_alpha_numeric_ratio, row.upper_to_all,
                                                             row.upper_to_lower, row.size_increment, row.comment_len,
                                                             row.vulgar_word_ratio, row.longest_Word, row.bad_words], group="editor", _class=row["class"])


        article_node = Node("Article", name=str(row.articleid), node_id=row.articleid, label="editor", feature=[row.size, row.word_count] + [0]*8, group="article", _class=3)
        graph.add_node(row.articleid, label=row.articletitle, feature=[row.size, row.word_count] + [0]*8, group="article", _class=3)
        if "category" in row and row["category"] is not np.nan:
            category_node = Node("Category", name=str(row.cat_code) , node_id=row.cat_code, label="editor", feature=[0] * 10, group="category", _class=4)
            category_edge = BELONGS_TO(article_node, category_node)#Relationship(article_node, "BELONGS_TO", category_node, label="category")
            graph.add_node(row.cat_code, label=row.category, feature=[0] * 10, group="category", _class=4)
            graph.add_edge(row.cat_code, row.articleid, label="category")

        editor_article_edge = EDITED_BY(editor, article_node, label="edit")
        graph.add_edge(row.editid, row.articleid, label="edit")
        try:
            tx.create(editor_article_edge)
            if category_edge is not None:
                tx.create(category_edge)
        except Exception as err:
            print(err)
        finally:
            if editor is not None:
                del editor
            if article_node is not None:
                del article_node
            if category_edge is not None:
                del category_edge
            if category_node is not None:
                del category_node
            if editor_article_edge is not None:
                del editor_article_edge
    tx.commit()
    return graph


if __name__ == "__main__":

    data = pd.read_csv("data/category_data_new.csv")
    data = data.fillna(0)
    if 'category' in data.columns:
        data.loc[data["category"] == 0, "category"] = "unknown"
        data["category"] = data["category"].astype("category")
        data["cat_code"] = data["category"].cat.codes
        print({category:cat_code for category, cat_code in zip(data["category"], data["category"].cat.codes)})
    if "character_distribution" in data.columns:
        data = data.drop(["character_distribution"], axis=1)
    data = data[~data.editid.isin(data.articleid)]
    print(data.info())
    graph = shortcircuit(data)
    g = StellarGraph.from_networkx(graph, node_features="feature")
    print(g.info())
    filename="data/graph.json"
    with open(filename, "w+") as fd:
        fd.write(json.dumps(json_graph.node_link_data(graph)))

    print("Network generation complete!!")
