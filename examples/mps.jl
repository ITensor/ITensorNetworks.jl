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

ψ̃ = sim(dag(ψ); sites=[])

@visualize ψ̃

ψψ = ⊗(ψ̃, ψ; new_dim_names=("bra", "ket"))

@visualize ψψ

# quasi-optimal contraction sequence
sequence = optimal_contraction_sequence(ψψ)

print_tree(sequence)

inner_res = contract(ψψ; sequence)[]

@show inner_res

sub = ψψ[[("bra", 1), ("ket", 1), ("bra", 2), ("ket", 2)]]

@visualize sub
