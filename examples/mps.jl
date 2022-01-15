using ITensors
using ITensorNetworks
using ITensorUnicodePlots

g = chain_lattice_graph(4)

s = siteinds("S=1/2", g)

tn = ITensorNetwork(s; link_space=3)

@visualize tn

tn′ = sim(dag(tn); sites=[])

@visualize tn′

inner_tn = tn ⊗ tn′

@visualize inner_tn
