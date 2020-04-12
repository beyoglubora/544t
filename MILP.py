from pulp import *
import pickle


def get_V2_c_star(G, optimal_candidate):
    return [i for i, v in enumerate(G.nodes) if G.nodes[v]["voting_preference"][1] == optimal_candidate]


def get_V1_c_star(G, optimal_candidate):
    return [i for i, v in enumerate(G.nodes) if G.nodes[v]["voting_preference"][0] == optimal_candidate]


def get_V2_c_star_intersection_V1_cj(C, V1_cj, V2_c_star):
    res = []
    V2_c_star_set = set(V2_c_star)
    [res.append(V2_c_star_set.intersection(set(V1_cj[cj]))) for cj in range(0, C)]
    return res


def get_V1_cj(G, C):
    res = []
    for cj in range(0, C):
        res.append([i for i, v in enumerate(G.nodes) if G.nodes[v]["voting_preference"][0] == cj])
    return res


def get_V2_cj(G, C):
    res = []
    for cj in range(0, C):
        res.append([i for i, v in enumerate(G.nodes) if G.nodes[v]["voting_preference"][1] == cj])
    return res


def get_V1_c_star_intersection_V2_cj(C, V2_cj, V1_c_star):
    res = []
    V1_c_star_set = set(V1_c_star)
    [res.append(V1_c_star_set.intersection(set(V2_cj[cj]))) for cj in range(0, C)]
    return res


def milp_mov_destructive(G, C, r, mc_samples, k, optimal_candidate):
    V1_c_star = get_V1_c_star(G, optimal_candidate)
    V2_cj = get_V2_cj(G, C)
    V1_c_star_intersection_V2_cj = get_V1_c_star_intersection_V2_cj(C, V2_cj, V1_c_star)
    V1_cj = get_V1_cj(G, C)
    nodes = range(0, len(G.nodes))
    samples = range(0, r)
    candidates = range(0, C)
    max_term = max([len(V1_cj[x]) for x in candidates])
    M = 10000000
    prob = LpProblem("Maximize_MOV", LpMaximize)

    # define variables
    s_v = LpVariable.dicts("seed", nodes, lowBound=0, upBound=1, cat="Binary")
    x_v = LpVariable.dicts("reachable", (samples, nodes), lowBound=0, upBound=1, cat="Binary")
    g_d = LpVariable.dicts("margin_change", (samples, candidates), lowBound=None, upBound=None, cat="Integer")
    m_d = LpVariable.dicts("overall_change", samples, lowBound=None, upBound=None, cat="Integer")
    z_i_j = LpVariable.dicts("max_change_in_margin", (samples, candidates), lowBound=0, upBound=1, cat="Binary")

    # define objective
    prob += (1 / len(samples)) * lpSum([m_d[i] for i in samples])

    # define constraints
    prob += lpSum(s_v[i] for i in nodes) <= k, "Seed Node Constraint"
    for s in range(0, len(mc_samples)):
        prob += lpSum(z_i_j[s][i] for i in candidates) >= 1, 'z constraint ' + str(s)
        for i in nodes:
            prob += x_v[s][i] <= lpSum(s_v[j] for j in mc_samples[s][i]["ancestors"]), "x_" + str(s) + "_" + str(i) + "constraint"
        for c in candidates:
            prob += g_d[s][c] <= lpSum(x_v[s][j] for j in V1_c_star) + lpSum(x_v[s][j] for j in V1_c_star_intersection_V2_cj[c]), "change in margin constraint " + str(s) + "_" + str(c)
        for c in candidates:
            #prob += m_d[s] <= g_d[s][c] + max_term - len(V1_cj[c]), "overall change in margin constraint" + str(s) + "_" + str(c)
            prob += m_d[s] <= g_d[s][c] - max_term + len(V1_cj[c]) + M*(1-z_i_j[s][c]), "overall change in margin constraint" + str(s) + "_" + str(c)

    print("Solving the LP")
    prob.writeLP("MaxMOV.lp")
    #prob.solve(pulp.PULP_CBC_CMD(maxSeconds=1000))
    prob.solve(pulp.CPLEX_PY(timeLimit=1000, epgap=.01))
    seed_nodes = []
    for v in prob.variables():
        if v.varValue == 1 and "seed" in v.name:
            num = int(v.name.split("_")[1])
            seed_nodes.append(num)
    print(seed_nodes)
    print(len(seed_nodes))
    print("Status:", LpStatus[prob.status])
    print(pulp.value(prob.objective))
    return pulp.value(prob.objective)


def milp_mov_constructive(G, C, r, mc_samples, k, optimal_candidate):
    nodes = range(0, len(G.nodes))
    samples = range(0, r)
    candidates = range(0, C)
    V2_c_star = get_V2_c_star(G, optimal_candidate)
    V1_cj = get_V1_cj(G, C)
    V2_c_star_intersection_V1_cj = get_V2_c_star_intersection_V1_cj(C, V1_cj, V2_c_star)
    max_term = max([len(V1_cj[x]) for x in candidates])

    prob = LpProblem("Maximize_MOV", LpMaximize)

    # define variables
    s_v = LpVariable.dicts("seed", nodes, lowBound=0, upBound=1, cat="Binary")
    x_v = LpVariable.dicts("reachable", (samples, nodes), lowBound=0, upBound=1, cat="Binary")
    g_c = LpVariable.dicts("margin_change", (samples, candidates), lowBound=None, upBound=None, cat="Integer")
    m_c = LpVariable.dicts("overall_change", samples, lowBound=None, upBound=None, cat="Integer")

    # define objective
    prob += (1/len(samples))*lpSum([m_c[i] for i in samples])

    # define constraints
    prob += lpSum(s_v[i] for i in nodes) <= k, "Seed Node Constraint"
    for s in range(0, len(mc_samples)):
        for i in nodes:
            prob += x_v[s][i] <= lpSum(s_v[j] for j in mc_samples[s][i]["ancestors"]), "x_" + str(s) + "_" + str(i) + "constraint"
        for c in candidates:
            prob += g_c[s][c] <= lpSum(x_v[s][j] for j in V2_c_star) + lpSum(x_v[s][j] for j in V2_c_star_intersection_V1_cj[c]), "change in margin constraint " + str(s) + "_" + str(c)
        for c in candidates:
            prob += m_c[s] <= g_c[s][c] + max_term - len(V1_cj[c]), "overall change in margin constraint" + str(s) + "_" + str(c)

    print("Solving the LP")
    prob.writeLP("MaxMOV.lp")
    prob.solve(CPLEX_PY(timeLimit=1000, epgap=.01))
    seed_nodes = []
    for v in prob.variables():
        if v.varValue == 1 and "seed" in v.name:
            num = int(v.name.split("_")[1])
            seed_nodes.append(num)
    print(seed_nodes)
    print(len(seed_nodes))
    print("Status:", LpStatus[prob.status])
    print(pulp.value(prob.objective))
    return pulp.value(prob.objective)


def compute_milp(network, is_constructive, C, instance, r, k, optimal_candidate):
    with open('544tFinalProjectData/' + network + '/' + ('constructive' if is_constructive else 'destructive') + '/C_' + str(C) + '/graph_' + str(C) + '_' + str(instance) +'.pkl', 'rb') as f:
        G = pickle.load(f)

    with open('544tFinalProjectData/' + network + '/' + ('constructive' if is_constructive else 'destructive') + '/C_' + str(C) + '/compressed_mc_samples_' + str(C) + '_' + str(instance) +'.pkl', 'rb') as f:
        mc_samples = pickle.load(f)

    mc_samples = mc_samples[0:r]
    if is_constructive:
        return milp_mov_constructive(G, C, r, mc_samples, k, optimal_candidate)
    else:
        return milp_mov_destructive(G, C, r, mc_samples, k, optimal_candidate)



# '''
# SET PARAMETER
# '''
# r = 1000
# k = 25
# optimal_candidate = 0
# C = 10
#
# '''
# READ GRAPH
# You will need to change the file path to point
# to your local copy of the network.
# '''
# # with open('polblogs_graph_trimmed_' + str(C) + '.pkl', 'rb') as f:
# #     G = pickle.load(f)
# with open('./544tFinalProjectData/netscience/constructive/C_' + str(C) + '/netscience_graph_' + str(C) + '.pkl', 'rb') as f:
#    G = pickle.load(f)
#
# '''
# LOAD MC SAMPLES
# '''
# # with open('compressed_mc_samples_trimmed_' + str(C) + '.pkl', 'rb') as f:
# #     mc_samples = pickle.load(f)
# with open('./544tFinalProjectData/netscience/constructive/C_' + str(C) + '/compressed_mc_samples_' + str(C) + '.pkl', 'rb') as f:
#    mc_samples = pickle.load(f)
#
# '''
# TRIM SAMPLES
# '''
# #mc_samples = mc_samples[0:r]
#
# '''
# SOLVE MILP
# '''
# milp_mov_constructive()
# #milp_mov_destructive()
