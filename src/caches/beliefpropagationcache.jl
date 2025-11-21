using Graphs: IsDirected
using SplitApplyCombine: group
using LinearAlgebra: diag, dot
using ITensors: dir
using NamedGraphs.PartitionedGraphs:
    AbstractPartitionedGraph,
    PartitionedGraphs,
    PartitionedGraph,
    QuotientVertex,
    boundary_quotientedges,
    partitioned_vertices,
    quotient_graph,
    quotientedges,
    quotientvertices,
    unpartitioned_graph
using SimpleTraits: SimpleTraits, Not, @traitfn
using NDTensors: NDTensors, Algorithm

function default_cache_construction_kwargs(alg::Algorithm"bp", ψ::AbstractITensorNetwork)
    return (; partitioned_vertices = default_partitioned_vertices(ψ))
end

function default_cache_construction_kwargs(alg::Algorithm"bp", pg::PartitionedGraph)
    return (;)
end

struct BeliefPropagationCache{V, PV, PTN <: AbstractPartitionedGraph{V, PV}, MTS} <:
    AbstractBeliefPropagationCache{V, PV}
    partitioned_tensornetwork::PTN
    messages::MTS
end

#Constructors...
function BeliefPropagationCache(ptn::PartitionedGraph; messages = default_messages(ptn))
    return BeliefPropagationCache(ptn, messages)
end

function BeliefPropagationCache(tn::AbstractITensorNetwork, partitioned_vertices; kwargs...)
    ptn = PartitionedGraph(tn, partitioned_vertices)
    return BeliefPropagationCache(ptn; kwargs...)
end

function BeliefPropagationCache(
        tn::AbstractITensorNetwork;
        partitioned_vertices = default_partitioned_vertices(tn),
        kwargs...,
    )
    return BeliefPropagationCache(tn, partitioned_vertices; kwargs...)
end

function cache(alg::Algorithm"bp", tn; kwargs...)
    return BeliefPropagationCache(tn; kwargs...)
end

function partitioned_tensornetwork(bp_cache::BeliefPropagationCache)
    return bp_cache.partitioned_tensornetwork
end

messages(bp_cache::BeliefPropagationCache) = bp_cache.messages

function default_message(bp_cache::BeliefPropagationCache, edge::QuotientEdge)
    return default_message(datatype(bp_cache), linkinds(bp_cache, edge))
end

function Base.copy(bp_cache::BeliefPropagationCache)
    return BeliefPropagationCache(
        copy(partitioned_tensornetwork(bp_cache)), copy(messages(bp_cache))
    )
end

default_update_alg(bp_cache::BeliefPropagationCache) = "bp"
default_message_update_alg(bp_cache::BeliefPropagationCache) = "contract"
default_normalize(::Algorithm"contract") = true
default_sequence_alg(::Algorithm"contract") = "optimal"
function set_default_kwargs(alg::Algorithm"contract")
    normalize = get(alg.kwargs, :normalize, default_normalize(alg))
    sequence_alg = get(alg.kwargs, :sequence_alg, default_sequence_alg(alg))
    return Algorithm("contract"; normalize, sequence_alg)
end
function set_default_kwargs(alg::Algorithm"adapt_update")
    _alg = set_default_kwargs(get(alg.kwargs, :alg, Algorithm("contract")))
    return Algorithm("adapt_update"; adapt = alg.kwargs.adapt, alg = _alg)
end
default_verbose(::Algorithm"bp") = false
default_tol(::Algorithm"bp") = nothing
function set_default_kwargs(alg::Algorithm"bp", bp_cache::BeliefPropagationCache)
    verbose = get(alg.kwargs, :verbose, default_verbose(alg))
    maxiter = get(alg.kwargs, :maxiter, default_bp_maxiter(bp_cache))
    edge_sequence = get(alg.kwargs, :edge_sequence, default_bp_edge_sequence(bp_cache))
    tol = get(alg.kwargs, :tol, default_tol(alg))
    message_update_alg = set_default_kwargs(
        get(alg.kwargs, :message_update_alg, Algorithm(default_message_update_alg(bp_cache)))
    )
    return Algorithm("bp"; verbose, maxiter, edge_sequence, tol, message_update_alg)
end

function default_bp_maxiter(bp_cache::BeliefPropagationCache)
    return default_bp_maxiter(quotient_graph(bp_cache))
end
function default_bp_edge_sequence(bp_cache::BeliefPropagationCache)
    return default_edge_sequence(partitioned_tensornetwork(bp_cache))
end

Base.setindex!(bpc::BeliefPropagationCache, factor::ITensor, vertex) = not_implemented()
partitions(bpc::BeliefPropagationCache) = quotientvertices(partitioned_tensornetwork(bpc))
function PartitionedGraphs.quotientedges(bpc::BeliefPropagationCache)
    return quotientedges(partitioned_tensornetwork(bpc))
end
function PartitionedGraphs.partitioned_vertices(bpc::BeliefPropagationCache)
    return partitioned_vertices(partitioned_tensornetwork(bpc))
end

function environment(bpc::BeliefPropagationCache, verts::Vector; kwargs...)
    partition_verts = quotientvertices(bpc, verts)
    messages = incoming_messages(bpc, partition_verts; kwargs...)
    central_tensors = factors(bpc, setdiff(vertices(bpc, partition_verts), verts))
    return vcat(messages, central_tensors)
end

function region_scalar(bp_cache::BeliefPropagationCache, pv::QuotientVertex)
    incoming_mts = incoming_messages(bp_cache, [pv])
    local_state = factors(bp_cache, pv)
    ts = vcat(incoming_mts, local_state)
    sequence = contraction_sequence(ts; alg = "optimal")
    return contract(ts; sequence)[]
end

function region_scalar(bp_cache::BeliefPropagationCache, pe::QuotientEdge)
    ts = vcat(message(bp_cache, pe), message(bp_cache, reverse(pe)))
    sequence = contraction_sequence(ts; alg = "optimal")
    return contract(ts; sequence)[]
end

function rescale_messages(bp_cache::BeliefPropagationCache, pes)
    bp_cache = copy(bp_cache)
    mts = messages(bp_cache)
    for pe in pes
        me, mer = normalize.(mts[pe]), normalize.(mts[reverse(pe)])
        set!(mts, pe, me)
        set!(mts, reverse(pe), mer)
        n = region_scalar(bp_cache, pe)
        if isreal(n)
            me[1] *= sign(n)
            n *= sign(n)
        end

        sf = inv(sqrt(n))^inv(oftype(n, length(me)))
        set!(mts, pe, sf .* me)
        set!(mts, reverse(pe), sf .* mer)
    end
    return bp_cache
end
