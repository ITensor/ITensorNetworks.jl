@eval module $(gensym())
using DataGraphs: DataGraph, underlying_graph, vertex_data
using Graphs: add_vertex!, vertices
# Trigger package extension.
using GraphsFlows: GraphsFlows
using ITensors: Index, ITensor, contract, noncommoninds, random_itensor
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
using TensorOperations: TensorOperations
using Test: @test, @testset

@testset "test _binary_tree_partition_inds of a 2D network" begin
    N = (3, 3, 3)
    linkdim = 2
    rng = StableRNG(1234)
    network = random_tensornetwork(rng, IndsNetwork(named_grid(N)); link_space = linkdim)
    tn = Array{ITensor, length(N)}(undef, N...)
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
    par = _partition(tn, binary_tree_structure(tn); alg = "mincut_recursive_bisection")
    network = mapreduce(v -> collect(eachtensor(par[v])), vcat, vertices(par))
    @test isapprox(contract(tn), contract(network))
end
end
