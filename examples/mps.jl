using AbstractTrees
using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using Random

Random.seed!(1234)

g = chain_lattice_graph(4)

s = siteinds("S=1/2", g)

ψ = ITensorNetwork(s; link_space=2)

# randomize
randn!.(vertex_data(ψ))

@visualize ψ

is = IndsNetwork(ψ)
v = vertex_data(is)
e = edge_data(is)

ψ′ = sim(dag(ψ); sites=[])

@visualize ψ′

inner_ψ = ψ ⊗ ψ′

@visualize inner_ψ

# quasi-optimal contraction sequence
sequence = optimal_contraction_sequence(inner_ψ)

print_tree(sequence)

inner_res = contract(inner_ψ; sequence)[]

# not yet implemented
#sub_ψ = inner_ψ[[1, 2, 5, 6]]
