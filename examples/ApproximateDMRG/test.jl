using NamedGraphs
using Graphs
using ITensorNetworks: construct_underlying_forests

g = named_grid((4, 4))

@show length(edges(g))

gs = construct_underlying_forests(g)

@show [length(connected_components(gv)) for gv in gs]
