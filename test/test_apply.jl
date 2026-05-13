using Compat: Compat
using Graphs: vertices
using ITensorNetworks: BeliefPropagationCache, apply, environment, initialize_cache,
    norm_sqr_network, siteinds, update
using ITensors: ITensors, Algorithm, ITensor, inner, op
using NamedGraphs.NamedGraphGenerators: named_grid
using SplitApplyCombine: group
using StableRNGs: StableRNG
using TensorOperations: TensorOperations
using Test: @test, @test_throws, @testset
include("utils.jl")
@testset "apply" begin
    g_dims = (2, 2)
    g = named_grid(g_dims)
    s = siteinds("S=1/2", g)
    χ = 2
    rng = StableRNG(1234)
    ψ = random_tensornetwork(rng, s; link_space = χ)
    v1, v2 = (2, 2), (1, 2)
    ψψ = norm_sqr_network(ψ)
    # Vertices of `[v1, v2]` across all layers of `ψψ` (bra/ket/operator),
    # so the environment around them is just the incoming BP messages —
    # the per-site operator tensors aren't pulled in as central tensors.
    env_verts(vs) = [
        (v, suffix) for v in vs for suffix in ("bra", "ket", "operator")
    ]
    # Simple Belief Propagation grouping (one bra/ket/operator triple per
    # partition) gives a product environment around `[v1, v2]`, which is
    # what `apply` requires.
    pv_SBP = group(v -> v[1], vertices(ψψ))
    bp_cache = update(
        initialize_cache(Algorithm("bp"), ψψ; partitioned_vertices = pv_SBP);
        maxiter = 20
    )
    envsSBP = environment(bp_cache, env_verts((v1, v2)))
    # Column-grouping (one whole column per partition) gives a non-product
    # environment; `apply` should reject it.
    pv_col = group(v -> v[1][1], vertices(ψψ))
    bp_cache_col = update(
        initialize_cache(Algorithm("bp"), ψψ; partitioned_vertices = pv_col);
        maxiter = 20
    )
    envsGBP = environment(bp_cache_col, env_verts((v1, v2)))
    inner_alg = "exact"
    ngates = 5
    truncerr = 0.0
    singular_values = ITensor()
    function callback(; singular_values, truncation_error)
        truncerr = truncation_error
        return singular_values = singular_values
    end
    for i in 1:ngates
        o = op("RandomUnitary", s[v1]..., s[v2]...)
        ψOexact = apply(o, ψ; cutoff = nothing)
        ψOSBP = apply(o, ψ; envs = envsSBP, maxdim = χ, normalize = true, callback)
        fSBP =
            inner(ψOSBP, ψOexact; alg = inner_alg) /
            sqrt(
            inner(ψOexact, ψOexact; alg = inner_alg) *
                inner(ψOSBP, ψOSBP; alg = inner_alg)
        )
        @test !iszero(truncerr)
        @test real(fSBP * conj(fSBP)) > 0
        @test_throws ErrorException apply(o, ψ; envs = envsGBP, maxdim = χ)
    end
end
