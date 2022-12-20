using NamedGraphs
using ITensors
using ITensorNetworks
using ITensorUnicodePlots

g = named_grid((3, 5))
s = siteinds("S=1/2", g)
ψ = ITensorNetwork(s; link_space=4)
@visualize ψ
@show center(ψ)
@show periphery(ψ)
t = dijkstra_tree(ψ, only(center(ψ)))
@visualize t
@show a_star(ψ, (2, 1), (2, 5))
@show mincut_partitions(ψ)
@show mincut_partitions(ψ, (1, 1), (3, 5))
@show subgraphs(ψ, 2)
