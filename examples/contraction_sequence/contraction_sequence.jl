using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using UnicodePlots
using Random

Random.seed!(1234)

ITensors.disable_warn_order()

dims = (2, 3)
g = named_grid(dims)
s = siteinds("S=1/2", g)

χ = 10
ψ = randomITensorNetwork(s; link_space=χ)

tn = norm_network(ψ)

# Contraction sequence for exactly computing expectation values
# contract_edges = map(t -> (1, t...), collect(keys(cartesian_to_linear(dims))))
# inner_sequence = reduce((x, y) -> [x, y], contract_edges)

println("optimal")
seq_optimal = @time contraction_sequence(tn; alg="optimal")

using OMEinsumContractionOrders

println("greedy")
seq_greedy = @time contraction_sequence(tn; alg="greedy")

println("tree_sa")
seq_tree_sa = @time contraction_sequence(tn; alg="tree_sa")

println("sa_bipartite")
seq_sa_bipartite = @time contraction_sequence(tn; alg="sa_bipartite")

println("kahypar_bipartite")
using KaHyPar
seq_kahypar_bipartite = @time contraction_sequence(tn; alg="kahypar_bipartite", sc_target=200)
