using AbstractTrees
using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using Random

Random.seed!(1234)

g = chain_lattice_graph(4)

s = siteinds("S=1/2", g)

tn = ITensorNetwork(s; link_space=2)

# randomize
randn!.(vertex_data(tn))

@visualize tn

tn′ = sim(dag(tn); sites=[])

@visualize tn′

inner_tn = tn ⊗ tn′

@visualize inner_tn

# quasi-optimal contraction sequence
sequence = optimal_contraction_sequence(inner_tn)

print_tree(sequence)

inner_res = contract(inner_tn; sequence)[]

# not yet implemented
#sub_tn = inner_tn[[1, 2, 5, 6]]
