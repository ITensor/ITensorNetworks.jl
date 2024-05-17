@eval module $(gensym())
using DataGraphs: DataGraph, underlying_graph, vertex_data
using Graphs: add_vertex!, vertices
# Trigger package extension.
using GraphsFlows: GraphsFlows
using ITensors: Index, ITensor, contract, noncommoninds, random_itensor
using ITensorMPS: MPS
using ITensorNetworks:
  _DensityMartrixAlgGraph,
  _contract_deltas_ignore_leaf_partitions,
  _mps_partition_inds_order,
  _mincut_partitions,
  _partition,
  _rem_vertex!,
  IndsNetwork,
  ITensorNetwork,
  binary_tree_structure,
  eachtensor,
  path_graph_structure,
  random_tensornetwork
using NamedGraphs: NamedEdge, NamedGraph
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.GraphsExtensions:
  is_binary_arborescence, post_order_dfs_vertices, root_vertex
using OMEinsumContractionOrders: OMEinsumContractionOrders
using StableRNGs: StableRNG
using Test: @test, @testset

@testset "test mincut functions on top of MPS" begin
  i = Index(2, "i")
  j = Index(2, "j")
  k = Index(2, "k")
  l = Index(2, "l")
  m = Index(2, "m")
  n = Index(2, "n")
  o = Index(2, "o")
  p = Index(2, "p")

  T = random_itensor(i, j, k, l, m, n, o, p)
  M = MPS(T, (i, j, k, l, m, n, o, p); cutoff=1e-5, maxdim=500)
  tn = ITensorNetwork(M[:])
  for out in [binary_tree_structure(tn), path_graph_structure(tn)]
    @test out isa DataGraph
    @test is_binary_arborescence(out)
    @test length(vertex_data(out).values) == 8
  end
  out = _mps_partition_inds_order(tn, [o, p, i, j, k, l, m, n])
  @test out in [[i, j, k, l, m, n, o, p], [p, o, n, m, l, k, j, i]]
  p1, p2 = _mincut_partitions(tn, [k, l], [m, n])
  # When MPS bond dimensions are large, the partition will not across internal inds
  @test (length(p1) == 0) || (length(p2) == 0)

  M = MPS(T, (i, j, k, l, m, n, o, p); cutoff=1e-5, maxdim=2)
  tn = ITensorNetwork(M[:])
  p1, p2 = _mincut_partitions(tn, [k, l], [m, n])
  # When MPS bond dimensions are small, the partition will across internal inds
  @test sort(p1) == [1, 2, 3, 4]
  @test sort(p2) == [5, 6, 7, 8]
end

@testset "test _binary_tree_partition_inds of a 2D network" begin
  N = (3, 3, 3)
  linkdim = 2
  rng = StableRNG(1234)
  network = random_tensornetwork(rng, IndsNetwork(named_grid(N)); link_space=linkdim)
  tn = Array{ITensor,length(N)}(undef, N...)
  for v in vertices(network)
    tn[v...] = network[v...]
  end
  tn = ITensorNetwork(vec(tn[:, :, 1]))
  for out in [binary_tree_structure(tn), path_graph_structure(tn)]
    @test out isa DataGraph
    @test is_binary_arborescence(out)
    @test length(vertex_data(out).values) == 9
  end
end

@testset "test partition with mincut_recursive_bisection alg of disconnected tn" begin
  inds = [Index(2, "$i") for i in 1:5]
  tn = ITensorNetwork([random_itensor(i) for i in inds])
  par = _partition(tn, binary_tree_structure(tn); alg="mincut_recursive_bisection")
  network = mapreduce(v -> collect(eachtensor(par[v])), vcat, vertices(par))
  @test isapprox(contract(tn), contract(network))
end

@testset "test partition with mincut_recursive_bisection alg and approx_itensornetwork" begin
  i = Index(2, "i")
  j = Index(2, "j")
  k = Index(2, "k")
  l = Index(2, "l")
  m = Index(2, "m")
  for dtype in (Float64, Complex{Float64})
    T = random_itensor(dtype, i, j, k, l, m)
    M = MPS(T, (i, j, k, l, m); cutoff=1e-5, maxdim=5)
    network = M[:]
    out1 = contract(network...)
    tn = ITensorNetwork(network)
    inds_btree = binary_tree_structure(tn)
    par = _partition(tn, inds_btree; alg="mincut_recursive_bisection")
    par = _contract_deltas_ignore_leaf_partitions(par; root=root_vertex(inds_btree))
    networks = map(v -> par[v], vertices(par))
    network2 = reduce(union, networks)
    out2 = contract(network2)
    @test isapprox(out1, out2)
    # test approx_itensornetwork (here we call `contract` to test the interface)
    for structure in [path_graph_structure, binary_tree_structure]
      for alg in ["density_matrix", "ttn_svd"]
        approx_tn, lognorm = contract(
          tn;
          alg,
          output_structure=structure,
          contraction_sequence_kwargs=(; alg="sa_bipartite"),
        )
        network3 = collect(eachtensor(approx_tn))
        out3 = contract(network3...) * exp(lognorm)
        i1 = noncommoninds(network...)
        i3 = noncommoninds(network3...)
        @test (length(i1) == length(i3))
        @test isapprox(out1, out3)
      end
    end
  end
end

@testset "test caching in approx_itensornetwork" begin
  i = Index(2, "i")
  j = Index(2, "j")
  k = Index(2, "k")
  l = Index(2, "l")
  m = Index(2, "m")
  T = random_itensor(i, j, k, l, m)
  M = MPS(T, (i, j, k, l, m); cutoff=1e-5, maxdim=5)
  tn = ITensorNetwork(M[:])
  out_tree = path_graph_structure(tn)
  input_partition = _partition(tn, out_tree; alg="mincut_recursive_bisection")
  underlying_tree = underlying_graph(input_partition)
  # Change type of each partition[v] since they will be updated
  # with potential data type chage.
  p = DataGraph(NamedGraph())
  for v in vertices(input_partition)
    add_vertex!(p, v)
    p[v] = ITensorNetwork{Any}(input_partition[v])
  end
  alg_graph = _DensityMartrixAlgGraph(p, underlying_tree, root_vertex(out_tree))
  path = post_order_dfs_vertices(underlying_tree, root_vertex(out_tree))
  for v in path[1:2]
    _rem_vertex!(
      alg_graph,
      v;
      cutoff=1e-15,
      maxdim=10000,
      contraction_sequence_kwargs=(; alg="optimal"),
    )
  end
  # Check that a specific density matrix info has been cached
  @test haskey(alg_graph.caches.es_to_pdm, Set([NamedEdge(nothing, path[3])]))
end
end
