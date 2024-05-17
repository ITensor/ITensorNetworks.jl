@eval module $(gensym())
using Graphs: dfs_tree, nv, vertices
# Trigger package extension.
using GraphsFlows: GraphsFlows
using ITensors: Index, ITensor, delta, noncommoninds, random_itensor
using ITensorNetworks:
  IndsNetwork,
  ITensorNetwork,
  _contract_deltas,
  _contract_deltas_ignore_leaf_partitions,
  _noncommoninds,
  _partition,
  binary_tree_structure,
  eachtensor,
  flatten_siteinds,
  path_graph_structure,
  random_tensornetwork
using NamedGraphs.GraphsExtensions: leaf_vertices, root_vertex
using NamedGraphs.NamedGraphGenerators: named_grid
using StableRNGs: StableRNG
using Test: @test, @testset

@testset "test _contract_deltas with no deltas" begin
  i = Index(2, "i")
  t = random_itensor(i)
  tn = _contract_deltas(ITensorNetwork([t]))
  @test tn[1] == t
end

@testset "test _contract_deltas over ITensorNetwork" begin
  is = [Index(2, string(i)) for i in 1:6]
  a = ITensor(is[1], is[2])
  b = ITensor(is[2], is[3])
  delta1 = delta(is[3], is[4])
  delta2 = delta(is[5], is[6])
  tn = ITensorNetwork([a, b, delta1, delta2])
  tn2 = _contract_deltas(tn)
  @test nv(tn2) == 3
  @test issetequal(flatten_siteinds(tn), flatten_siteinds(tn2))
end

@testset "test _contract_deltas over partition" begin
  N = (3, 3, 3)
  linkdim = 2
  rng = StableRNG(1234)
  network = random_tensornetwork(rng, IndsNetwork(named_grid(N)); link_space=linkdim)
  tn = Array{ITensor,length(N)}(undef, N...)
  for v in vertices(network)
    tn[v...] = network[v...]
  end
  tn = ITensorNetwork(vec(tn[:, :, 1]))
  for inds_tree in [binary_tree_structure(tn), path_graph_structure(tn)]
    par = _partition(tn, inds_tree; alg="mincut_recursive_bisection")
    root = root_vertex(inds_tree)
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
end
