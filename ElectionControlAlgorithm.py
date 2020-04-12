import networkx as nx
import DataPreprocessing
import LazyGreedy
import MILP
import numpy as np

networks = ['synthetic']
is_constructive_params = [True]
instances = 1
num_mc_samples = 100
r = 100
C = [2]
k = [2]
optimal_candidate = 0


res = {}
for network in networks:
    if network in ["polblogs", "netscience", "synthetic"]:
        G = nx.read_gml(network + ".gml")
    else:
        if network == "irvine":
            G = nx.read_adjlist(network + ".txt", comments="%")
        else:
            G = nx.read_adjlist(network, comments="%")
    for is_constructive in is_constructive_params:
        for candidate in C:
            print("Working on " + network + " with C = " + str(candidate))
            for num_seed_nodes in k:
                print("Evalutating k = " + str(num_seed_nodes))
                perc_matches = []
                for instance in range(0, instances):
                    #DataPreprocessing.preprocess_data(G.copy(), candidate, p, num_mc_samples, r, optimal_candidate, instance, is_constructive, network)
                    algo_val = LazyGreedy.lazy_greedy(network, num_mc_samples, candidate, is_constructive, num_seed_nodes, instance, optimal_candidate)
                    milp_val = MILP.compute_milp(network, is_constructive, candidate, instance, r, num_seed_nodes, optimal_candidate)
                    perc_matches.append(min(algo_val/milp_val, 1))
                    print(algo_val/milp_val)
                res[network + "C" + str(candidate) + "k" + str(num_seed_nodes) + ("constructive" if is_constructive else "destructive")] = np.mean(perc_matches)
                print(res)


