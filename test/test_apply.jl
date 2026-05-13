using Compat: Compat
using Dictionaries: Dictionary, set!
using Graphs: dst, src, vertices
using ITensorNetworks: BeliefPropagationCache, apply, environment, identity_messages,
    inner_network, siteinds, update
using ITensors: ITensors, ITensor, Index, commoninds, dag, inner, op, prime
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.PartitionedGraphs:
    PartitionedGraph, QuotientEdge, partitioned_vertices, quotientedges
using SplitApplyCombine: group
using StableRNGs: StableRNG
using TensorOperations: TensorOperations
using Test: @test, @test_throws, @testset
include("utils.jl")

# Build identity-style initial messages on each quotient edge of an LFN
# (where bra = `dag(prime(ket))`) by pairing each ket Index `k` with its
# bra counterpart `dag(prime(k))` directly, so the construction is
# robust to multiple link indices per edge.
function _lfn_identity_messages(ψψ, ptn::PartitionedGraph)
    pairings = Dictionary{QuotientEdge, Pair{Vector{Index}, Vector{Index}}}()
    pv = partitioned_vertices(ptn)
    for pe in quotientedges(ptn)
        src_orig = unique(first.(filter(v -> last(v) == "ket", pv[parent(src(pe))])))
        dst_orig = unique(first.(filter(v -> last(v) == "ket", pv[parent(dst(pe))])))
        for (from_orig, to_orig, e) in (
                (src_orig, dst_orig, pe),
                (dst_orig, src_orig, reverse(pe)),
            )
            kets = Index[]
            for v_from in from_orig, v_to in to_orig
                append!(kets, commoninds(ψψ[(v_from, "ket")], ψψ[(v_to, "ket")]))
            end
            bras = dag.(prime.(kets))
            set!(pairings, e, bras => kets)
        end
    end
    return identity_messages(ITensors.scalartype(ψψ), pairings)
end

@testset "apply" begin
    g_dims = (2, 2)
    g = named_grid(g_dims)
    s = siteinds("S=1/2", g)
    χ = 2
    rng = StableRNG(1234)
    ψ = random_tensornetwork(rng, s; link_space = χ)
    v1, v2 = (2, 2), (1, 2)
    ψψ = inner_network(ψ, ψ)
    # Simple Belief Propagation grouping (one bra+ket per partition) gives
    # a product environment around `[v1, v2]`, which is what `apply` requires.
    ptn_SBP = PartitionedGraph(ψψ, group(v -> v[1], vertices(ψψ)))
    bp_cache =
        BeliefPropagationCache(ptn_SBP; messages = _lfn_identity_messages(ψψ, ptn_SBP))
    bp_cache = update(bp_cache; maxiter = 20)
    envsSBP = environment(bp_cache, [(v1, "bra"), (v1, "ket"), (v2, "bra"), (v2, "ket")])
    # Column-grouping (one whole column per partition) gives a non-product
    # environment; `apply` should reject it.
    ptn_col = PartitionedGraph(ψψ, group(v -> v[1][1], vertices(ψψ)))
    bp_cache_col =
        BeliefPropagationCache(ptn_col; messages = _lfn_identity_messages(ψψ, ptn_col))
    bp_cache_col = update(bp_cache_col; maxiter = 20)
    envsGBP = environment(
        bp_cache_col, [(v1, "bra"), (v1, "ket"), (v2, "bra"), (v2, "ket")]
    )
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
