import pulp
import networkx as nx
import os
import matplotlib.pyplot as plt
import json
import time
import pandas as pd

def solve(path="/content/21SS_CSPTP-baesed_ILP/dataset/nsfnetbw"):
    filename = os.path.join(path, "network.gml")
    physical_network = nx.read_gml(filename, label="id")

    filename = os.path.join(path, "service_chain_requirements.json")
    with open(filename) as f:
        connections = json.load(f)
    for connection in connections:
        connection['required_processing_func'] = {int(k): v for k, v in connection['required_processing_func'].items()}

    filename = os.path.join(path, "function_placement.json")
    with open(filename) as f:
        placement = function_placement["placement"]
    F = set(function_placement['functions'])

    imaginary_links = pd.DataFrame.from_dict(placement)
    imaginary_nodes = {f: f+physical_node_size for f in F}

    G = physical_network.copy()
    for func, node in imaginary_nodes.items():
        G.add_node(node, lng=-(func+1)*8 - 70, lat=60, delay=0)
    imaginary_links.apply(lambda x: G.add_edge(x.physical_node, imaginary_nodes[x.function], delay=0), axis=1)
    imaginary_links.apply(lambda x: G.add_edge(imaginary_nodes[x.function], x.physical_node, delay=x.delay * 1e-3), axis=1) # d_{i,j}^{func}

    a = []
    b = []
    for c in connections:
        R_c = c['request']
        K_c = len(R_c)
        o_c = c['origin']
        d_c = c['destination']
        a_ck = []
        b_ck = []
        for k in range(K_c+1):
            if k == 0:
                a_ck.append(o_c)
            if k == K_c:
                b_ck.append(d_c)
            else:
                a_ck.append(imaginary_nodes[R_c[k]])
                b_ck.append(imaginary_nodes[R_c[k]])
        a.append(a_ck)
        b.append(b_ck)

    # formualte ILP
    m = pulp.LpProblem("SPTP-based-ILP", pulp.LpMinimize)

    x = {}
    for c, connection in enumerate(connections):
        K_c = len(connection['request'])
        for k in range(K_c+1):
            for i, j in G.edges():
                x[i,j,c,k] = pulp.LpVariable("x_{:},{:},{:},{:}".format(i, j, c, k), 0, 1, pulp.LpBinary)

    m += pulp.lpSum(G[i][j]["delay"] * pulp.lpSum(x[i,j,c,k]           for c in range(len(connections)) for k in range(len(connections[c]['request'])+1) ) for i, j in physical_network.edges()) \
            + pulp.lpSum(G.nodes()[v]["delay"] * pulp.lpSum(x[v,j,c,k] for c in range(len(connections)) for k in range(len(connections[c]['request'])+1) ) for v, j in physical_network.edges()) \
            + pulp.lpSum(G[v_f][v]["delay"] * pulp.lpSum(x[v_f,v,c,k]  for c in range(len(connections)) for k in range(len(connections[c]['request'])+1) ) for v_f, v in G.edges() if v_f in imaginary_nodes.values()), "Objective function"

    for c, connection in enumerate(connections):
        K_c = len(connection['request'])
        for k in range(K_c + 1):
            for i in G.nodes():
                if i == a[c][k]:
                    m += pulp.lpSum(x[i, j, c, k] for j in G.neighbors(i)) == 1, f"flow_constraint_c{c}_k{k}_i{i}"
                elif i == b[c][k]:
                    m += pulp.lpSum(x[j, i, c, k] for j in G.neighbors(i) ) == 1, f"flow_constraint_c{c}_k{k}_i{i}"
                else:
                    m += pulp.lpSum(x[i, j, c, k] for j in G.neighbors(i)) == pulp.lpSum(x[j, i, c, k] for j in G.neighbors(i)), f"flow_constraint_c{c}_k{k}_i{i}"

    for c, connection in enumerate(connections):
        K_c = len(connection['request'])
        for k in range(K_c):
            v_f_ck = b[c][k]
            for i in G.neighbors(v_f_ck):
                m += x[i, v_f_ck, c, k] == x[v_f_ck, i, c, k+1], "connectivity_of_subpath_constraint_i{:}j{:}c{:}k{:}".format(i, v_f_ck, c, k)

    for c, connection in enumerate(connections):
        K_c = len(connection['request'])
        for k in range(K_c+1):
            for i, v_f in G.edges():
                if v_f in imaginary_nodes.values() and not v_f is b[c][k]:
                    m += x[i, v_f, c, k] == 0, "infeasible_link_i{:}j{:}k{:}c{:}".format(i, v_f, k, c)

    for i, j in physical_network.edges():
        m += pulp.lpSum(connection['required_bandwidth'] * pulp.lpSum(x[i, j, c, k] for k in range(len(connection['request'])+1)) for c, connection in enumerate(connections)) <= physical_network[i][j]["bandwidth"], "physical_link_capacity_i{:}j{:}".format(i, j)

    for v in physical_network.nodes():
        m += pulp.lpSum(connection["required_processing_node"] * pulp.lpSum(x[v,j,c,k] for j in physical_network.neighbors(v) for k in range(len(connection['request'])+1)) + \
                pulp.lpSum(connection["required_processing_func"][v_f-physical_node_size] * pulp.lpSum(x[v_f,v,c,k] for k in range(len(connection['request'])+1)) for v_f in G.neighbors(v) if v_f in imaginary_nodes.values() and v_f-physical_node_size in connection['request']) for c, connection in enumerate(connections)) \
                <= physical_network.nodes()[v]["capacity"], "processing_capacity_constraint_v{:}".format(v)

    for c, connection in enumerate(connections):
        K_c = len(connection['request'])
        o = connection['origin']
        d = connection['destination']
        m += pulp.lpSum(x[i,o,c,0] for i in G.neighbors(o)) == 0, "constraint_7_w{:}".format(c)
        m += pulp.lpSum(x[d,i,c,len(connection['request'])] for i in G.neighbors(d)) == 0, "constraint_8_w{:}".format(c)

    start = time.perf_counter()
    status = m.solve()
    end = time.perf_counter()
    elapsed = end - start

    return pulp.LpStatus[status], pulp.value(m.objective), elapsed
