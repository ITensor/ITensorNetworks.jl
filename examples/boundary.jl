using NamedGraphs
using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using Metis

tn = ITensorNetwork(named_grid((6, 3)); link_space=4)

@visualize tn

ptn = PartitionedGraph(tn; nvertices_per_partition=2)
sub_vs_1 = vertices(ptn, PartitionVertex(1))
sub_vs_2 = vertices(ptn, PartitionVertex(2))

@show (1, 1) ∈ sub_vs_1
@show (6, 3) ∈ sub_vs_2

@show boundary_edges(tn, sub_vs_1)
@show boundary_vertices(tn, sub_vs_1)
@show inner_boundary_vertices(tn, sub_vs_1)
@show outer_boundary_vertices(tn, sub_vs_1)
