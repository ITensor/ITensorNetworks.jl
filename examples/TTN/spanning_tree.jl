using ITensors
using ITensorNetworks
using ITensorGLMakie

s = siteinds("S=1/2", named_grid((4, 4)))
ψ = ITensorNetwork(s; link_space=3)

@visualize ψ

readline()

# Gives a comb pattern
t = bfs_tree(ψ, (1, 1))
@visualize t

readline()

# Gives a snake pattern
t_dfs = dfs_tree(ψ, (1, 1))
@visualize t_dfs

nothing
