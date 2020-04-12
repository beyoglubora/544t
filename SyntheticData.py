import networkx as nx

G_synthetic = nx.DiGraph()
G_synthetic.add_nodes_from(list(range(0, 63)))
big_node = 60
minor_node1 = 61
minor_node2 = 62
big_node_edge_set1 = [(big_node, node) for node in list(G_synthetic.nodes)[0:25]]
big_node_edge_set2 = [(big_node, node) for node in list(G_synthetic.nodes)[35:60]]
G_synthetic.add_edges_from(big_node_edge_set1)
G_synthetic.add_edges_from(big_node_edge_set2)
minor_node1_edge_set = [(minor_node1, node) for node in list(G_synthetic.nodes)[0:30]]
minor_node2_edge_set = [(minor_node2, node) for node in list(G_synthetic.nodes)[30:60]]
G_synthetic.add_edges_from(minor_node1_edge_set)
G_synthetic.add_edges_from(minor_node2_edge_set)
nx.write_gml(G_synthetic, "synthetic.gml")

