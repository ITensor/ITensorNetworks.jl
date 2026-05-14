using Dictionaries: Dictionary
using Graphs: is_tree
using ITensors.NDTensors: @Algorithm_str, Algorithm
using NamedGraphs.PartitionedGraphs: PartitionedGraph, quotient_graph

# Build a cache for algorithm `alg` on `tn`. The fallback constructs a
# plain `BeliefPropagationCache` with no message defaults; the
# `QuadraticFormNetwork` specialization injects `identity_messages` on
# loopy quotient graphs (canonical for the structurally ψ-vs-ψ case).
function initialize_cache(alg::Algorithm"bp", tn; kwargs...)
    return BeliefPropagationCache(tn; kwargs...)
end

function initialize_cache(
        alg::Algorithm"bp",
        fn::QuadraticFormNetwork;
        partitioned_vertices = default_partitioned_vertices(fn),
        messages = nothing
    )
    ptn = PartitionedGraph(fn, partitioned_vertices)
    if isnothing(messages)
        messages = if is_tree(quotient_graph(ptn))
            Dictionary()
        else
            identity_messages(fn; partitioned_vertices)
        end
    end
    return BeliefPropagationCache(ptn; messages)
end
