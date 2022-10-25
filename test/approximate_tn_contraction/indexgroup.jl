using ITensors
using ITensorNetworks.ApproximateTNContraction:
  IndexGroup, get_index_groups, get_leaves, neighbor_index_groups
using ITensorNetworks.ApproximateTNContraction:
  inds_network,
  line_network,
  IndexAdjacencyTree,
  topo_sort,
  get_ancestors,
  generate_adjacency_tree,
  minswap_adjacency_tree!,
  minswap_adjacency_tree,
  approximate_contract

# @testset "test generate_adjacency_tree" begin
#   N = (3, 3)
#   tn_inds = inds_network(N...; linkdims=2, periodic=false)
#   tn = vec(map(inds -> randomITensor(inds...), tn_inds))
#   ctree = line_network(tn)
#   tn_leaves = get_leaves(ctree)
#   ctrees = topo_sort(ctree; leaves=tn_leaves)
#   ctree_to_igs = Dict{Vector,Vector{IndexGroup}}()
#   index_groups = get_index_groups(ctree)
#   for c in vcat(tn_leaves, ctrees)
#     ctree_to_igs[c] = neighbor_index_groups(c, index_groups)
#   end
#   ctree_to_ancestors = get_ancestors(ctree)
#   adj_tree1 = generate_adjacency_tree(
#     tn_leaves[4], ctree_to_ancestors[tn_leaves[4]], ctree_to_igs
#   )
#   adj_tree2 = generate_adjacency_tree(
#     ctrees[2], ctree_to_ancestors[ctrees[2]], ctree_to_igs
#   )
#   for adj_tree in [adj_tree1, adj_tree1]
#     @test length(adj_tree.children) == 3
#     @test adj_tree.fixed_order = true
#     c1, c2, c3 = adj_tree.children
#     @test length(c1.children) == 1
#     @test length(c2.children) == 2
#     @test length(c3.children) == 1
#   end
# end

# @testset "test minswap_adjacency_tree!" begin
#   i = IndexGroup([Index(2, "i")])
#   j = IndexGroup([Index(3, "j")])
#   k = IndexGroup([Index(2, "k")])
#   l = IndexGroup([Index(4, "l")])
#   m = IndexGroup([Index(5, "m")])
#   n = IndexGroup([Index(5, "n")])
#   I = IndexAdjacencyTree(i)
#   J = IndexAdjacencyTree(j)
#   K = IndexAdjacencyTree(k)
#   L = IndexAdjacencyTree(l)
#   M = IndexAdjacencyTree(m)
#   N = IndexAdjacencyTree(n)
#   JKL = IndexAdjacencyTree([J, K, L], false, false)
#   tree = IndexAdjacencyTree([I, JKL, M], false, false)
#   tree_copy = copy(tree)
#   tree2 = IndexAdjacencyTree([i, k, m, j, l], true, true)
#   nswaps = minswap_adjacency_tree!(tree, tree2)
#   @test nswaps == 1
#   @test tree.children == [i, m, k, j, l]
#   @test tree.fixed_direction && tree.fixed_order
#   # test minswap_adjacency_tree
#   tree3 = IndexAdjacencyTree([i, k, n, m], true, true)
#   tree4 = IndexAdjacencyTree([l, n, j], true, true)
#   out = minswap_adjacency_tree(tree_copy, tree3, tree4)
#   @test out.children in [[i, m, k, j, l], [i, m, k, l, j], [m, i, k, j, l], [m, i, k, l, j]]
# end

@testset "test approximate_contract" begin
  N = (4, 4)
  tn_inds = inds_network(N...; linkdims=2, periodic=false)
  tn = vec(map(inds -> randomITensor(inds...), tn_inds))
  ctree = line_network(tn)
  approximate_contract(ctree; cutoff=1e-5, maxdim=20)
end
