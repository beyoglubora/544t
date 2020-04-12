import networkx as nx
import LazyGreedy
import MILP
import numpy as np
import DataPreprocessing


networks = ['synthetic']
is_constructive_params = [True]
instances = 1
num_mc_samples = 1
r = 1
p = .1
C = [2]
k = [2]
optimal_candidate = 0
for network in networks:
    if network in ["polblogs", "netscience", "synthetic"]:
        G = nx.read_gml(network + ".gml")
    elif network == "synthetic":
        G = nx.read_gml(network + ".gml")
    else:
        if network == "irvine":
            G = nx.read_adjlist(network + ".txt", comments="%")
        else:
            G = nx.read_adjlist(network, comments="%")
    for is_constructive in is_constructive_params:
        for candidate in C:
            print("Analysis on " + network + " with C = " + str(candidate))
            for num_seed_nodes in k:
                print("Evalutating k = " + str(num_seed_nodes))
                for instance in range(0, instances):
                    centrality, avg_centrality = LazyGreedy.analysis(network, num_mc_samples, candidate, is_constructive, num_seed_nodes, instance, optimal_candidate)
                    print(centrality)
                    print(avg_centrality)

