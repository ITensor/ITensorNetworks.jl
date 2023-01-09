using AbstractTrees
using NamedGraphs
using ITensors
using ITensorNetworks
using ITensorUnicodePlots

g = named_binary_tree(3)
s = siteinds("S=1/2", g)
ψ = TTNS(s; link_space=3)

for v in vertices(ψ)
  ψ[v] = randn!(ψ[v])
end

@visualize ψ

@show neighbors(ψ, (1,))
@show neighbors(ψ, (1, 1, 1))
@show incident_edges(ψ, (1, 1))
@show leaf_vertices(ψ)
@show is_leaf(ψ, (1,))
@show is_leaf(ψ, (1, 1, 1))

e = (1, 1) => (1,)
ψ̃ = contract(ψ, e)

@visualize ψ̃

ψᴴ = prime(dag(ψ); sites=[])
Z = ψᴴ ⊗ ψ;

@visualize Z

# Contract across bra and ket
for v in vertices(ψ)
  global Z = contract(Z, (v, 2) => (v, 1))
end

@visualize Z

sequence = contraction_sequence(Z)

@show sequence

z = contract(Z; sequence)[]

@show √z

# Contract according to a post-order depth-first
# search, inward towards the root vertex.
# https://en.wikipedia.org/wiki/Tree_traversal#Depth-first_search
z2 = Z
root_vertex = ((1,), 1)
@visualize z2
for e in post_order_dfs_edges(z2, root_vertex)
  @show e
  global z2 = contract(z2, e)
  @visualize z2
end
@show √(z2[root_vertex][1])

e = edgetype(ψ)((1,) => (1, 1))
ψ_svd = svd(ψ, e)
U = ψ_svd[src(e)]
S = ψ_svd[e, "S"]
V = ψ_svd[e, "V"]

@visualize ψ_svd

@show norm(U * S * V - ψ[src(e)])

ψ̃_svd = contract(ψ_svd, (e, "V") => dst(e))
ψ̃_svd = contract(ψ̃_svd, (e, "S") => dst(e))

@visualize ψ̃_svd

e = edgetype(ψ)((1,) => (1, 1))
ψ_qr = qr(ψ, e)
Q = ψ_qr[src(e)]
R = ψ_qr[e, "R"]

@visualize ψ_qr

@show norm(Q * R - ψ[src(e)])

ψ̃_qr = contract(ψ_qr, (e, "R") => dst(e))

@visualize ψ̃_qr

# Orthogonalize according to post-order
# depth-first search, towards the root vertex.
# https://en.wikipedia.org/wiki/Tree_traversal#Depth-first_search
ψ_ortho = ψ
root_vertex = (1, 1)
@visualize ψ_ortho

for e in post_order_dfs_edges(ψ_ortho, root_vertex)
  @show e
  global ψ_ortho = orthogonalize(ψ_ortho, e)
  @visualize ψ_ortho
end

@show √(
  contract(
    norm_sqr_network(ψ_ortho); sequence=contraction_sequence(norm_sqr_network(ψ_ortho))
  )[],
)
@show √(contract(norm_sqr_network(ψ); sequence=contraction_sequence(norm_sqr_network(ψ)))[])
@show norm(ψ_ortho[root_vertex])
@show √(inner(ψ, ψ))
@show √(inner(ψ_ortho, ψ_ortho))
@show norm(ψ)
@show norm(ψ_ortho)

ψ_ortho = orthogonalize(ψ, (1,))
@show norm(ψ_ortho)
@show norm(ψ_ortho[(1,)])

nothing
