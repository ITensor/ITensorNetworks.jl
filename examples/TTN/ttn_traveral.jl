using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using Graphs

g = binary_tree(3)
s = siteinds("S=1/2", g)
ψ = ITensorNetwork(s; link_space=2)

@visualize ψ

bfs_tree_ψ1 = bfs_tree(ψ, (1,))
dfs_tree_ψ1 = dfs_tree(ψ, (1,))

nothing
