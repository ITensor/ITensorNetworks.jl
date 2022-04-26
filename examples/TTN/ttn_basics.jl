using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using NamedGraphs

g = named_binary_tree(3)

@show g
@show g[1, 1, :]
@show g[1, 2, :]
@visualize g

s = siteinds("S=1/2", g)
ψ = ITensorNetwork(s; link_space=3)

@visualize ψ

bfs_tree_ψ = bfs_tree(ψ, 1, 2)
dfs_tree_ψ = dfs_tree(ψ, 1, 2)

nothing
