import networkx as nx
import DataPreprocessing

networks = ['netscience']
is_constructive_params = [True, False]
instances = 30
num_mc_samples = 100
r = 50
p = .1
C = [2, 5, 10]
k = [25, 50, 100]
optimal_candidate = 0
for network in networks:
    G = nx.read_gml(network + ".gml")
    for candidate in C:
        for is_constructive in is_constructive_params:
            for instance in range(0, instances):
                print("For network " + network + ": preprocessing C = " + str(candidate) + " instance = " + str(instance))
                DataPreprocessing.preprocess_data(G, candidate, p, num_mc_samples, r, optimal_candidate, instance, is_constructive, network)