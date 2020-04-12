import networkx as nx
import DataPreprocessing


networks = ['synthetic']
is_constructive_params = [True]
instances = 1
num_mc_samples = 100
r = 100
p = .1
C = [2]
optimal_candidate = 0
for network in networks:
    if network in ["polblogs", "netscience", "synthetic"]:
        G = nx.read_gml(network + ".gml")
    else:
        if network == "irvine":
            G = nx.read_adjlist(network + ".txt", comments="%")
        else:
            G = nx.read_adjlist(network, comments="%")
    for candidate in C:
        for is_constructive in is_constructive_params:
            for instance in range(0, instances):
                print("For network " + network + ": preprocessing C = " + str(candidate) + " instance = " + str(instance))
                DataPreprocessing.preprocess_data(G, candidate, p, num_mc_samples, r, optimal_candidate, instance, is_constructive, network, network=="synthetic")
