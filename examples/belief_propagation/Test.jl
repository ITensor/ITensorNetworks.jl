using ITensors
using ITensorNetworks
using ITensorNetworks: compute_message_tensors, nested_graph_leaf_vertices
using Random
using KaHyPar
using SplitApplyCombine

n = 4
dims = (n, n)
g = named_grid(dims)
s = siteinds("S=1/2", g)
chi = 2

Random.seed!(5467)

#bra
ψ = randomITensorNetwork(s; link_space=chi)
#bra-ket (but not actually contracted)
ψψ = ψ ⊗ prime(dag(ψ); sites=[])

#Site to take expectation value on
v = (1, 1)

#Now do Simple Belief Propagation to Measure Sz on Site v
nsites = 2

Z = partition(partition(ψψ, group(v -> v[1], vertices(ψψ))); nvertices_per_partition=nsites)

@show nested_graph_leaf_vertices(Z; toplevel=true)
