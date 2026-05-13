using Dictionaries: Dictionary
using Graphs: is_tree
using ITensors.NDTensors: @Algorithm_str, Algorithm
using ITensors: scalar
using LinearAlgebra: normalize
using NamedGraphs.PartitionedGraphs:
    AbstractPartitionedGraph, PartitionedGraph, quotient_graph

# Build a cache appropriate for `f` on `tn` using algorithm `alg`. The
# `f` tag carries the calling context (e.g. `scalar`, `normalize`,
# `expect`, `rescale`, `environment`) so per-purpose methods can inject
# context-specific initialization. The fallback constructs a plain
# `BeliefPropagationCache` with no message defaults.
function initialize_cache(f, alg::Algorithm"bp", tn::AbstractITensorNetwork; kwargs...)
    return BeliefPropagationCache(tn; kwargs...)
end

function initialize_cache(
        f, alg::Algorithm"bp", ptn::AbstractPartitionedGraph; kwargs...
    )
    return BeliefPropagationCache(ptn; kwargs...)
end

# Core helper: build a BPC on a form network with `identity_messages`
# on loopy quotient graphs (empty messages on trees). Used by the
# per-purpose specializations below where the form network is
# structurally ψ-vs-ψ, so `identity_messages(fn, ptn)` is canonical.
function _bp_cache_identity_messages(
        fn::AbstractFormNetwork;
        partitioned_vertices = default_partitioned_vertices(fn),
        messages = nothing
    )
    ptn = PartitionedGraph(fn, partitioned_vertices)
    if isnothing(messages)
        messages = is_tree(quotient_graph(ptn)) ? Dictionary() : identity_messages(fn, ptn)
    end
    return BeliefPropagationCache(ptn; messages)
end

function initialize_cache(
        ::typeof(scalar), alg::Algorithm"bp", fn::QuadraticFormNetwork; kwargs...
    )
    return _bp_cache_identity_messages(fn; kwargs...)
end

function initialize_cache(
        ::typeof(normalize), alg::Algorithm"bp", fn::AbstractFormNetwork; kwargs...
    )
    return _bp_cache_identity_messages(fn; kwargs...)
end

function initialize_cache(
        ::typeof(rescale), alg::Algorithm"bp", fn::AbstractFormNetwork; kwargs...
    )
    return _bp_cache_identity_messages(fn; kwargs...)
end

function initialize_cache(
        ::typeof(expect), alg::Algorithm"bp", fn::QuadraticFormNetwork; kwargs...
    )
    return _bp_cache_identity_messages(fn; kwargs...)
end
