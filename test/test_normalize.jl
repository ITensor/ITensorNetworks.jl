using Graphs: SimpleGraph, uniform_tree
using ITensorNetworks: BeliefPropagationCache, QuadraticFormNetwork,
    default_partitioned_vertices, edge_scalars, identity_messages, messages,
    norm_sqr_network, rescale, scalartype, siteinds, vertex_scalars
using ITensors: dag, inner, scalar
using LinearAlgebra: normalize
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using NamedGraphs.PartitionedGraphs: PartitionedGraph
using NamedGraphs: NamedGraph
using StableRNGs: StableRNG
using TensorOperations: TensorOperations
using Test: @test, @testset
include("utils.jl")
@testset "Normalize" begin

    #First lets do a flat tree
    nx, ny = 2, 3
    χ = 2
    rng = StableRNG(1234)

    g = named_comb_tree((nx, ny))
    tn = random_tensornetwork(rng, g; link_space = χ)

    tn_r = rescale(tn; alg = "exact")
    @test scalar(tn_r; alg = "exact") ≈ 1.0

    tn_r = rescale(tn; alg = "bp", cache_update_kwargs = (; maxiter = 20))
    @test scalar(tn_r; alg = "exact") ≈ 1.0

    #Now a state on a loopy graph
    Lx, Ly = 3, 2
    χ = 2
    rng = StableRNG(1234)

    g = named_grid((Lx, Ly))
    s = siteinds("S=1/2", g)
    x = random_tensornetwork(rng, ComplexF32, s; link_space = χ)

    ψ = normalize(x; alg = "exact")
    @test scalar(norm_sqr_network(ψ); alg = "exact") ≈ 1.0

    # Loopy BP needs explicit initial messages; the form-network's
    # `identity_messages` gives a properly-flowing pair-delta starting
    # point even for QN-graded indices.
    qfn = QuadraticFormNetwork(x)
    pv = default_partitioned_vertices(qfn)
    ptn = PartitionedGraph(qfn, pv)
    ψIψ_bpc = Ref(BeliefPropagationCache(ptn; messages = identity_messages(qfn, ptn)))
    ψ = normalize(
        x; alg = "bp", (cache!) = ψIψ_bpc, update_cache = true,
        cache_update_kwargs = (; maxiter = 20)
    )
    ψIψ_bpc = ψIψ_bpc[]
    @test all(m -> scalartype(only(m)) == ComplexF32, messages(ψIψ_bpc))
    @test all(x -> x ≈ 1.0, edge_scalars(ψIψ_bpc))
    @test all(x -> x ≈ 1.0, vertex_scalars(ψIψ_bpc))
    @test scalar(
        QuadraticFormNetwork(ψ);
        alg = "bp",
        cache_update_kwargs = (; maxiter = 20)
    ) ≈ 1.0
end
