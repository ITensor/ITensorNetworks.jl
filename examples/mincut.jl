using NamedGraphs
using ITensors
using ITensorNetworks
using ITensorUnicodePlots

g = named_grid(5)
s = siteinds("S=1/2", g)

ψ = ITensorNetwork(s; link_space=10)

# Or:

ss = ∪(dag(s), s'; merge_data=union)
ρ = ITensorNetwork(ss; link_space=2)

tn = ⊗(ρ', ρ, ψ)
tn_flattened = flatten_networks(ρ', ρ, ψ)
# tn = ρ' ⊗ ρ ⊗ ψ
@visualize tn

@show center(tn)

v = first(center(tn))

dijk_parents = dijkstra_parents(tn, v)
dijk_mst = dijkstra_mst(tn, v)
dijk_tree = dijkstra_tree(tn, v)

bfs_tree_tn = bfs_tree(tn, v)

@show eccentricity(tn, v)
@show radius(tn)
@show radius(tn)
@show diameter(tn)
@show periphery(tn)

s = dijk_tree
t = bfs_tree_tn
@visualize s
@visualize t

v1 = first(periphery(tn))
nds = neighborhood_dists(tn, v1, nv(tn))
d_and_i = findmax(vd -> vd[2], nds)
v2 = nds[d_and_i[2]][1]
@show v1, v2
p1, p2 = mincut_partitions(tn, v1, v2)
@show p1
@show p2

display(adjacency_matrix(tn_flattened))
tn_flattened_p = symrcm_permute(tn_flattened)
display(adjacency_matrix(tn_flattened_p))
