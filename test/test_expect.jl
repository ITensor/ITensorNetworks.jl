using Dictionaries: AbstractDictionary
using Graphs: SimpleGraph, uniform_tree
using ITensorNetworks: ITensorNetwork, expect, siteinds
using ITensors: Op
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs: NamedGraph, vertices
using SplitApplyCombine: group
using StableRNGs: StableRNG
using TensorOperations: TensorOperations
using Test: @test, @testset
include("utils.jl")

# Whole-network `expect` returns a dictionary, so compare the two algorithms vertex by
# vertex.
agree(a, b) = collect(keys(a)) == collect(keys(b)) && all(v -> a[v] ≈ b[v], keys(a))

@testset "Test Expect" begin
    @testset "Whole-network result is indexed by vertex" begin
        # Product state: BP is exact and the expected values are known analytically.
        g = named_grid((2, 2))
        s = siteinds("S=1/2", g)
        ψ = productstate(v -> isodd(sum(v)) ? "↑" : "↓", s)

        sz = expect(ψ, "Sz"; alg = "bp", cache_update_kwargs = (; maxiter = 20))
        @test sz isa AbstractDictionary
        @test collect(keys(sz)) == collect(vertices(ψ))
        # Index the result directly with a vertex of `ψ`.
        @test sz[(1, 1)] ≈ -0.5
        @test sz[(1, 2)] ≈ +0.5
        @test sz[(2, 1)] ≈ +0.5
        @test sz[(2, 2)] ≈ -0.5
        @test all(v -> sz[v] ≈ (isodd(sum(v)) ? +0.5 : -0.5), vertices(ψ))
    end

    @testset "Vertex subsets and single operators" begin
        L, χ = 4, 2
        g = NamedGraph(SimpleGraph(uniform_tree(L)))
        s = siteinds("S=1/2", g)
        rng = StableRNG(1234)
        ψ = random_tensornetwork(rng, s; link_space = χ)

        sz = expect(ψ, "Sz"; alg = "exact")
        # Reversed relative to `vertices(ψ)`, to pin down the ordering of the result.
        vs = reverse(collect(vertices(ψ)))

        # Passing an explicit `vertices` argument mirrors its container type: a `Vector`
        # of vertices in, a `Vector` of values in the same order out.
        sz_vec = expect(ψ, "Sz", vs; alg = "exact")
        @test sz_vec isa Vector
        @test sz_vec ≈ [sz[v] for v in vs]
        @test expect(ψ, "Sz", vs; alg = "bp") ≈ sz_vec

        # A subset works the same way.
        @test expect(ψ, "Sz", vs[1:2]; alg = "exact") ≈ sz_vec[1:2]

        # A single `Op` returns a bare number, not a collection.
        for v in vs
            sz_v = expect(ψ, Op("Sz", v); alg = "exact")
            @test sz_v isa Number
            @test sz_v ≈ sz[v]
        end
    end

    @testset "Tree: BP is exact" begin
        L, χ = 4, 2
        g = NamedGraph(SimpleGraph(uniform_tree(L)))
        s = siteinds("S=1/2", g)
        rng = StableRNG(1234)
        ψ = random_tensornetwork(rng, s; link_space = χ)
        @test agree(expect(ψ, "Sz"; alg = "bp"), expect(ψ, "Sz"; alg = "exact"))
    end

    @testset "Grid: BP grouped by column is exact" begin
        L, χ = 2, 2
        g = named_grid((L, L))
        s = siteinds("S=1/2", g)
        rng = StableRNG(1234)
        ψ = random_tensornetwork(rng, s; link_space = χ)
        quadratic_form_vertices = reduce(
            vcat, [[(v, "ket"), (v, "bra"), (v, "operator")] for v in vertices(ψ)]
        )
        cache_construction_kwargs = (;
            partitioned_vertices = group(v -> first(first(v)), quadratic_form_vertices),
        )
        sz_bp = expect(
            ψ, "Sz"; alg = "bp", cache_construction_kwargs,
            cache_update_kwargs = (; maxiter = 20)
        )
        @test agree(sz_bp, expect(ψ, "Sz"; alg = "exact"))
    end

    @testset "Quantum numbers" begin
        # Product state, so BP should be exact.
        L = 2
        g = named_grid((L, L))
        s = siteinds("S=1/2", g; conserve_qns = true)
        ψ = productstate(v -> isodd(sum(v)) ? "↑" : "↓", s)
        sz_bp = expect(ψ, "Sz"; alg = "bp", cache_update_kwargs = (; maxiter = 20))
        @test agree(sz_bp, expect(ψ, "Sz"; alg = "exact"))
    end
end
