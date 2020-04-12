import numpy as np
import heapq
import pickle


def change_in_margin_for_node_set_constructive(G, C, node_list, votes, nodes, optimal_candidate):
    changes_in_margin = []
    for candidate in range(0, C):
        g = 0
        #if candidate != optimal_candidate:
        for n in nodes:
            if G.nodes[node_list[n]]["voting_preference"][0] == candidate and G.nodes[node_list[n]]["voting_preference"][1] == optimal_candidate:
                g += 2
            elif G.nodes[node_list[n]]["voting_preference"][1] == optimal_candidate:
                g += 1
        changes_in_margin.append(g + np.max(votes) - votes[candidate])
    return min(changes_in_margin)


def change_in_margin_for_node_set_destructive(G, C, node_list, votes, nodes, optimal_candidate):
    changes_in_margin = []
    for candidate in range(0, C):
        g = 0
        for n in nodes:
            if G.nodes[node_list[n]]["voting_preference"][1] == candidate and \
                    G.nodes[node_list[n]]["voting_preference"][0] == optimal_candidate:
                g += 2
            elif G.nodes[node_list[n]]["voting_preference"][0] == optimal_candidate:
                g += 1
        changes_in_margin.append(g + votes[candidate])
    return max(changes_in_margin) - np.max(votes)


def compute_influence_for_node_set(G, C, node_list, votes, nodes, samples, is_constructive, optimal_candidate):
    vals = []
    for s in samples:
        reachable_nodes = []
        for node in nodes:
            reachable_nodes = np.union1d(reachable_nodes, s[node]["influenced_descendants"]).astype(int)
        val = change_in_margin_for_node_set_constructive(G, C, node_list, votes, reachable_nodes, optimal_candidate) if is_constructive else change_in_margin_for_node_set_destructive(G, C, node_list, votes, reachable_nodes, optimal_candidate)
        vals.append(val)
        # vals.append(len(reachable_nodes))
    return np.mean(vals)


def compute_influence_for_node(G, C, node_list, votes, node, samples, is_constructive, optimal_candidate):
    vals = []
    for s in samples:
        #vals.append(len(s[node]["influenced_descendants"]))
        val = change_in_margin_for_node_set_constructive(G, C, node_list, votes, s[node]["influenced_descendants"], optimal_candidate) if is_constructive else change_in_margin_for_node_set_destructive(G, C, node_list, votes, s[node]["influenced_descendants"], optimal_candidate)
        vals.append(val)
    return np.mean(vals)


def greedy_once(G, C, node_list, votes, nodes, samples, is_constructive, optimal_candidate):
    heap = []
    influences = []
    for n in nodes:
        influence = compute_influence_for_node(G, C, node_list, votes, n, samples, is_constructive, optimal_candidate)
        influences.append(influence)
        heapq.heappush(heap, (-influence, n))

    return heap, influences


def lazy_greedy(network, num_mc_samples, C, is_constructive, num_seed_nodes, instance, optimal_candidate):
    with open('544tFinalProjectData/' + network + '/' + ('constructive' if is_constructive else 'destructive') + '/C_' + str(C) + '/compressed_mc_samples_' + str(C) + '_' + str(instance) +'.pkl', 'rb') as f:
        samples = pickle.load(f)
    samples = samples[0:num_mc_samples]

    with open('544tFinalProjectData/' + network + '/' + ('constructive' if is_constructive else 'destructive') + '/C_' + str(C) + '/graph_' + str(C) + '_' + str(instance) +'.pkl', 'rb') as f:
        G = pickle.load(f)

    votes = []
    for candidate in range(0, C):
        votes.append(len([node for node in G.nodes if G.nodes[node]["voting_preference"][0] == candidate]))
    node_list = list(G.nodes)
    marginal_influences = []
    print("RUNNING LAZY GREEDY")
    heap, influences = greedy_once(G, C, node_list, votes, list(np.arange(0, len(G.nodes))), samples, is_constructive, optimal_candidate)
    with open('544tFinalProjectData/' + network + '/' + ('constructive' if is_constructive else 'destructive') + '/C_' + str(C) + '/influences_' + str(C) + '_' + str(instance) + '.pkl', 'wb') as f:
        pickle.dump(influences, f)
    influence, node = heapq.heappop(heap)
    prev_influence = -influence
    marginal_influences.append(prev_influence)
    seed_nodes = {node}
    seed_nodes_list = [node]
    while len(seed_nodes) < num_seed_nodes:
        print(seed_nodes)
        max_found = False
        while not max_found:
            influence, node = heapq.heappop(heap)
            influence = compute_influence_for_node_set(G, C, node_list, votes, (seed_nodes | {node}), samples, is_constructive, optimal_candidate)
            marginal_influence = influence - prev_influence
            heapq.heappush(heap, (-marginal_influence, node))
            max_found = heap[0][1] == node
        marginal_influence, node = heapq.heappop(heap)
        marginal_influences.append(-marginal_influence)
        prev_influence = prev_influence + (-marginal_influence)
        seed_nodes.add(node)
        seed_nodes_list.append(node)
    print(seed_nodes_list)
    print(prev_influence)
    with open('544tFinalProjectData/' + network + '/' + ('constructive' if is_constructive else 'destructive') + '/C_' + str(C) + '/seed_nodes_' + str(C) + '_' + str(instance) + '.pkl', 'wb') as f:
        pickle.dump(seed_nodes_list, f)
    with open('544tFinalProjectData/' + network + '/' + ('constructive' if is_constructive else 'destructive') + '/C_' + str(C) + '/marginal_influences_' + str(C) + '_' + str(instance) + '.pkl', 'wb') as f:
        pickle.dump(marginal_influences, f)
    return prev_influence


def analysis(network, num_mc_samples, C, is_constructive, num_seed_nodes, instance, optimal_candidate):
    seed_nodes = []
    with open('544tFinalProjectData/' + network + '/' + ('constructive' if is_constructive else 'destructive') + '/C_' + str(C) + '/seed_nodes_' + str(C) + '_' + str(instance) +'.pkl', 'rb') as f:
        seed_nodes = pickle.load(f)
    influences = []
    with open('544tFinalProjectData/' + network + '/' + ('constructive' if is_constructive else 'destructive') + '/C_' + str(C) + '/influences_' + str(C) + '_' + str(instance) + '.pkl', 'rb') as f:
        influences = pickle.load(f)
    with open('544tFinalProjectData/' + network + '/' + ('constructive' if is_constructive else 'destructive') + '/C_' + str(C) + '/marginal_influences_' + str(C) + '_' + str(instance) + '.pkl', 'rb') as f:
        marginal_influences = pickle.load(f)
    avg_influence = sum(influences) / len(influences)
    print(marginal_influences)
    seed_node_influences = []
    print(seed_nodes)
    for s in seed_nodes:
        seed_node_influences.append(influences[s])
    return seed_node_influences, avg_influence


''' 
SET PARAMETERS
'''
#k = 25
#C = 10
#optimal_candidate = 0

'''
LOAD GRAPH
'''
# with open('polblogs_graph_trimmed_' + str(C) + '.pkl', 'rb') as f:
#    G = pickle.load(f)
# with open('./544tFinalProjectData/netscience/constructive/C_' + str(C) + '/netscience_graph_' + str(C) + '.pkl', 'rb') as f:
#    G = pickle.load(f)
# votes = []
# for candidate in range(0, C):
#     votes.append(len([node for node in G.nodes if G.nodes[node]["voting_preference"][0] == candidate]))
# node_list = list(G.nodes)
# '''
# LOAD MC SAMPLES
# '''
# # with open('compressed_mc_samples_trimmed_' + str(C) + '.pkl', 'rb') as f:
# #    samples = pickle.load(f)
# with open('./544tFinalProjectData/netscience/constructive/C_' + str(C) + '/compressed_mc_samples_' + str(C) + '.pkl', 'rb') as f:
#    samples = pickle.load(f)
#
# #samples = samples[0:100]
#
# '''
# GREEDY
# Run greedy algorithm
# '''
# lazy_greedy(k)
