using ITensors
using ITensorNetworks
using ITensorUnicodePlots

g = named_binary_tree(3)
s = siteinds("S=1/2", g)
ψ = TTNS(s; link_space=3)

@visualize ψ

bfs_tree_ψ = bfs_tree(ψ, 1, 2)
dfs_tree_ψ = dfs_tree(ψ, 1, 2)

e = 1 => (1, 1)
ψ̃ = contract(ψ, e)

@visualize ψ̃

nothing
