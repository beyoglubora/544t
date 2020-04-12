from networkx import *
import random
import numpy as np
import pickle


def index_nodes(G):
    for i, v in enumerate(G.nodes):
        G.nodes[v]["index"] = i


def assign_random_voting_preferences(G, C):
    for v in G.nodes:
        candidates = list(np.arange(C))
        random.shuffle(candidates)
        G.nodes[v]["voting_preference"] = candidates


def assign_voting_preferences(G, C):
    for v in G.nodes:
        candidates = list(np.arange(C))[::-1]
        G.nodes[v]["voting_preference"] = candidates


def create_mc_samples(G, p, num_mc_samples, r, optimal_candidate, is_constructive):
    mc_samples = []
    for i in np.arange(0, num_mc_samples):
        sample = create_empty_copy(G)
        edges = np.asarray(G.edges)
        random_edges = edges[np.where(np.random.rand(len(edges)) < p)]
        sample.add_edges_from(random_edges)
        mc_sample = []
        for node in sample:
            # changing the value after voting_preference key here is what determines constructive
            # from destructive for the algorithm
            desc = [G.nodes[v]["index"] for v in descendants(sample, node) if G.nodes[v]["voting_preference"][1 if is_constructive else 0] == optimal_candidate]
            anc = []
            if r > 0:
                anc = [G.nodes[v]["index"] for v in ancestors(sample, node)]
            mc_sample.append({"influenced_descendants": desc, "ancestors": anc})
        mc_samples.append(mc_sample)
        r -= 1
    return mc_samples


def preprocess_data(G, C, p, num_mc_samples, r, optimal_candidate, instance, is_constructive, network, is_synthetic = False):
    index_nodes(G)
    if is_synthetic:
        assign_voting_preferences(G, C)
    else:
        assign_random_voting_preferences(G, C)
    mc_samples = create_mc_samples(G, p, num_mc_samples, r, optimal_candidate, is_constructive)

    with open('544tFinalProjectData/' + network + '/' + ('constructive' if is_constructive else 'destructive') + '/C_' + str(C) + '/compressed_mc_samples_' + str(C) + '_' + str(instance) +'.pkl', 'wb') as f:
        pickle.dump(mc_samples, f)

    with open('544tFinalProjectData/' + network + '/' + ('constructive' if is_constructive else 'destructive') + '/C_' + str(C) + '/graph_' + str(C) + '_' + str(instance) +'.pkl', 'wb') as f:
        pickle.dump(G, f)




'''
READ GRAPH
You will need to change the file path to point
to your local copy of the network.
'''
# G = nx.read_gml("netscience.gml")
# with open('polblogs_graph_trimmed_' + "2" + '.pkl', 'rb') as f:
#     G = pickle.load(f)

'''
SET PRE-PROCESSING PARAMETERS
'''
# C = 10
# r = 1000
# num_mc_samples = 1000
# optimal_candidate = 0
# p = .1

'''
INDEX NODES
'''
# index_nodes()

'''
ASSIGN VOTING PREFERENCES RANDOMLY
'''
# assign_random_voting_preferences()


'''
CREATE COMPRESSED MONTE CARLO SAMPLES
Compressed graph to only contain necessary data
for greedy + MILP.
'''
# samples = create_mc_samples(r)

'''
SAVE COMPRESSED MC SAMPLES
'''
# with open('544tFinalProjectData/netscience/constructive/C_' + str(C) + '/compressed_mc_samples_' + str(C) + '.pkl', 'wb') as f:
#     pickle.dump(samples, f)

'''
SAVE GRAPH
'''
# with open('544tFinalProjectData/netscience/constructive/C_' + str(C) + '/netscience_graph_' + str(C) + '.pkl', 'wb') as f:
#     pickle.dump(G, f)

