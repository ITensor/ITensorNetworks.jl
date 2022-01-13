using ITensors
using ITensorNetworks

g = chain_lattice_graph(4)

# TODO: errors, fix
tn = ITensorNetwork(g; link_space=3)
