using ITensors
using ITensorNetworks:
  _contract_deltas, _contract_deltas_ignore_leaf_partitions, _noncommoninds, _root

@testset "test _contract_deltas over ITensorNetwork" begin
  is = [Index(2, string(i)) for i in 1:6]
  a = ITensor(is[1], is[2])
  b = ITensor(is[2], is[3])
  delta1 = delta(is[3], is[4])
  delta2 = delta(is[5], is[6])
  tn = ITensorNetwork([a, b, delta1, delta2])
  tn2 = _contract_deltas(tn)
  @test nv(tn2) == 3
  @test Set(noncommoninds(Vector{ITensor}(tn)...)) ==
    Set(noncommoninds(Vector{ITensor}(tn2)...))
end

@testset "test _contract_deltas over partition" begin
  N = (3, 3, 3)
  linkdim = 2
  network = randomITensorNetwork(IndsNetwork(named_grid(N)); link_space=linkdim)
  tn = Array{ITensor,length(N)}(undef, N...)
  for v in vertices(network)
    tn[v...] = network[v...]
  end
  tn = ITensorNetwork(vec(tn[:, :, 1]))
  for inds_tree in [binary_tree_structure(tn), path_graph_structure(tn)]
    par = partition(tn, inds_tree; alg="mincut_recursive_bisection")
    root = _root(inds_tree)
    par_contract_deltas = _contract_deltas_ignore_leaf_partitions(par; root=root)
    @test Set(_noncommoninds(par)) == Set(_noncommoninds(par_contract_deltas))
    leaves = leaf_vertices(dfs_tree(par_contract_deltas, root))
    nonleaf_vertices = setdiff(vertices(par_contract_deltas), leaves)
    nvs = sum([nv(par_contract_deltas[v]) for v in nonleaf_vertices])
    # all delta tensors in nonleaf vertives should be contracted, so the
    # remaining tensors are all non-delta tensors
    @test nvs == 9
  end
end
