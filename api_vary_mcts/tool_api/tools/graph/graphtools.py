import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os

class graph_toolkits():
    # init
    def __init__(self, path):
        self.path = path

        with open('{}/data/external_corpus/dblp/paper_net.pkl'.format(self.path), 'rb') as f:
            self.paper_net = pickle.load(f)

        with open('{}/data/external_corpus/dblp/author_net.pkl'.format(self.path), 'rb') as f:
            self.author_net = pickle.load(f)
        
        with open("{}/data/external_corpus/dblp/title2id_dict.pkl".format(self.path), "rb") as f:
            self.title2id_dict = pickle.load(f)
        with open("{}/data/external_corpus/dblp/author2id_dict.pkl".format(self.path), "rb") as f:
            self.author2id_dict = pickle.load(f)
        with open("{}/data/external_corpus/dblp/id2title_dict.pkl".format(self.path), "rb") as f:
            self.id2title_dict = pickle.load(f)
        with open("{}/data/external_corpus/dblp/id2author_dict.pkl".format(self.path), "rb") as f:
            self.id2author_dict = pickle.load(f)
    
    def load_graph(self, graph_name):
        # print(graph_name)
        if graph_name == 'dblp':
            
            return "DBLP data is loaded, including two graphs: AuthorNet and PaperNet."
        else:
            return f"We don't have the graph data named {graph_name}"
    
    def check_neighbours(self, graph, node):
        # graph, node = argument.split(', ')
        if graph == 'PaperNet':
            graph = self.paper_net
            dictionary = self.title2id_dict
            inv_dict = self.id2title_dict
        elif graph == 'AuthorNet':
            graph = self.author_net
            dictionary = self.author2id_dict
            inv_dict = self.id2author_dict
        neighbour_list = []
        for neighbour in graph.neighbors(dictionary[node]):
            neighbour_list.append(inv_dict[neighbour])
        return str(neighbour_list)

    # check the attributes of the nodes
    def check_nodes(self, graph, node):
        # graph, node = argument.split(', ')
        if graph == 'PaperNet':
            graph = self.paper_net
            dictionary = self.title2id_dict
            inv_dict = self.id2title_dict
        elif graph == 'AuthorNet':
            graph = self.author_net
            dictionary = self.author2id_dict
            inv_dict = self.id2author_dict
        return str(graph.nodes[dictionary[node]])

    # check the attributes of the edges
    def check_edges(self, graph, node1, node2):
        # graph, node1, node2 = argument.split(', ')
        if graph == 'PaperNet':
            graph = self.paper_net
            dictionary = self.title2id_dict
            inv_dict = self.id2title_dict
            edge = graph.edges[dictionary[node1], dictionary[node2]]
            return str(edge)
        elif graph == 'AuthorNet':
            graph = self.author_net
            dictionary = self.author2id_dict
            inv_dict = self.id2title_dict
            edge = graph.edges[dictionary[node1], dictionary[node2]]
            for id in range(len(edge['papers'])):
                edge['papers'][id] = inv_dict[edge['papers'][id]]
            return str(edge)

if __name__ == '__main__':
    # test
    graph_toolkits = graph_toolkits("/yinxr/workhome/zzhong/chenguoxin/datasets/ToolQA")
    logs = graph_toolkits.load_graph('dblp')
    print(str(graph_toolkits.check_neighbours('PaperNet, HRFormer: High-Resolution Vision Transformer for Dense Predict.')))
    print(str(graph_toolkits.check_neighbours('AuthorNet, Chao Zhang')))
    print(str(graph_toolkits.check_nodes('PaperNet, Learning the Principle of Least Action with Reinforcement Learning.')))
    print(str(graph_toolkits.check_nodes('AuthorNet, He Zhang')))
    # print(graph_toolkits.check_edges('PaperNet, 5fbe62d191e011e6e11b3d73, 5fbe62d191e011e6e11b3d73'))
    print(str(graph_toolkits.check_edges('AuthorNet, Chao Zhang, Weihong Lin')))
    print(str(graph_toolkits.check_neighbours('AuthorNet, Weihong Lin')))
