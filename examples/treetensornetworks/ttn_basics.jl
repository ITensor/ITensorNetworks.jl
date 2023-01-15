using NamedGraphs
using ITensors
using ITensorNetworks
using ITensorUnicodePlots

g = named_binary_tree(3)

@show g
filter_vertices(v, v1, v2) = length(v) ≥ 2 && v[1] == v1 && v[2] == v2
@show subgraph(v -> filter_vertices(v, 1, 1), g)
@show subgraph(v -> filter_vertices(v, 1, 2), g)
@visualize g

s = siteinds("S=1/2", g)
ψ = ITensorNetwork(s; link_space=3)

@visualize ψ

bfs_tree_ψ = bfs_tree(ψ, (1, 2))
dfs_tree_ψ = dfs_tree(ψ, (1, 2))

nothing
