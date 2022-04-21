using Graphs
using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using Suppressor

dims = (4, 4)
g = hypercubic_lattice_graph(dims)

s = siteinds("S=1/2", g)

χ = 5
ψ = ITensorNetwork(s; link_space=χ)

@visualize ψ edge_labels = (; plevs=true)

ψ′ = ψ'
@visualize ψ′ edge_labels = (; plevs=true)

v = (2, 2)
neighbor_edges = [v => nv for nv in neighbors(ψ, v)]
@show siteinds(ψ, v)
@show [e => linkinds(ψ, e) for e in neighbor_edges]

npartitions = 4
partitions = partition(ψ, npartitions)

@show partitions

nothing
