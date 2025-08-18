using Adapt: Adapt, adapt, adapt_structure
using Graphs: Graphs, IsDirected
using SplitApplyCombine: group
using LinearAlgebra: diag, dot
using ITensors: dir
using NamedGraphs.PartitionedGraphs:
  PartitionedGraphs,
  PartitionedGraph,
  PartitionVertex,
  boundary_partitionedges,
  partitionvertices,
  partitionedges,
  unpartitioned_graph
using SimpleTraits: SimpleTraits, Not, @traitfn
using NamedGraphs.SimilarType: SimilarType
using NDTensors: NDTensors

abstract type AbstractBeliefPropagationCache{V,PV} <: AbstractITensorNetwork{V} end

function SimilarType.similar_type(bpc::AbstractBeliefPropagationCache)
  return typeof(tensornetwork(bpc))
end
function data_graph_type(bpc::AbstractBeliefPropagationCache)
  return data_graph_type(tensornetwork(bpc))
end
data_graph(bpc::AbstractBeliefPropagationCache) = data_graph(tensornetwork(bpc))

function message_update(alg::Algorithm"contract", contract_list::Vector{ITensor};)
  sequence = contraction_sequence(contract_list; alg=alg.kwargs.sequence_alg)
  updated_messages = contract(contract_list; sequence)
  message_norm = norm(updated_messages)
  if alg.kwargs.normalize && !iszero(message_norm)
    updated_messages /= message_norm
  end
  return ITensor[updated_messages]
end

function message_update(alg::Algorithm"adapt_update", contract_list::Vector{ITensor};)
  adapted_contract_list = alg.kwargs.adapt.(contract_list)
  updated_messages = message_update(alg.kwargs.alg, adapted_contract_list)
  dtype = datatype(first(contract_list))
  return map(adapt(dtype), updated_messages)
end

#TODO: Take `dot` without precontracting the messages to allow scaling to more complex messages
function message_diff(message_a::Vector{ITensor}, message_b::Vector{ITensor})
  lhs, rhs = contract(message_a), contract(message_b)
  f = abs2(dot(lhs / norm(lhs), rhs / norm(rhs)))
  return 1 - f
end

function default_message(datatype::Type{<:AbstractArray}, inds_e)
  return [adapt(datatype, denseblocks(delta(i))) for i in inds_e]
end

function default_message(elt::Type{<:Number}, inds_e)
  return default_message(Vector{elt}, inds_e)
end
default_messages(ptn::PartitionedGraph) = Dictionary()
@traitfn default_bp_maxiter(g::::(!IsDirected)) = is_tree(g) ? 1 : 30
@traitfn function default_bp_maxiter(g::::IsDirected)
  return default_bp_maxiter(undirected_graph(underlying_graph(g)))
end
default_partitioned_vertices(ψ::AbstractITensorNetwork) = group(v -> v, vertices(ψ))

function Base.setindex!(bpc::AbstractBeliefPropagationCache, factor::ITensor, vertex)
  return not_implemented()
end
partitioned_tensornetwork(bpc::AbstractBeliefPropagationCache) = not_implemented()
messages(bpc::AbstractBeliefPropagationCache) = not_implemented()
function default_message(
  bpc::AbstractBeliefPropagationCache, edge::PartitionEdge; kwargs...
)
  return not_implemented()
end
default_update_alg(bpc::AbstractBeliefPropagationCache) = not_implemented()
default_message_update_alg(bpc::AbstractBeliefPropagationCache) = not_implemented()
Base.copy(bpc::AbstractBeliefPropagationCache) = not_implemented()
default_bp_maxiter(alg::Algorithm, bpc::AbstractBeliefPropagationCache) = not_implemented()
function default_edge_sequence(alg::Algorithm, bpc::AbstractBeliefPropagationCache)
  return not_implemented()
end
function environment(bpc::AbstractBeliefPropagationCache, verts::Vector; kwargs...)
  return not_implemented()
end
function region_scalar(bpc::AbstractBeliefPropagationCache, pv::PartitionVertex; kwargs...)
  return not_implemented()
end
function region_scalar(bpc::AbstractBeliefPropagationCache, pe::PartitionEdge; kwargs...)
  return not_implemented()
end
partitions(bpc::AbstractBeliefPropagationCache) = not_implemented()
PartitionedGraphs.partitionedges(bpc::AbstractBeliefPropagationCache) = not_implemented()

default_bp_edge_sequence(bpc::AbstractBeliefPropagationCache) = not_implemented()
default_bp_maxiter(bpc::AbstractBeliefPropagationCache) = not_implemented()

function tensornetwork(bpc::AbstractBeliefPropagationCache)
  return unpartitioned_graph(partitioned_tensornetwork(bpc))
end

function factors(bpc::AbstractBeliefPropagationCache, verts::Vector)
  return ITensor[bpc[v] for v in verts]
end

function factors(
  bpc::AbstractBeliefPropagationCache, partition_verts::Vector{<:PartitionVertex}
)
  return factors(bpc, vertices(bpc, partition_verts))
end

function factors(bpc::AbstractBeliefPropagationCache, partition_vertex::PartitionVertex)
  return factors(bpc, [partition_vertex])
end

function vertex_scalars(bpc::AbstractBeliefPropagationCache, pvs=partitions(bpc); kwargs...)
  return map(pv -> region_scalar(bpc, pv; kwargs...), pvs)
end

function edge_scalars(
  bpc::AbstractBeliefPropagationCache, pes=partitionedges(bpc); kwargs...
)
  return map(pe -> region_scalar(bpc, pe; kwargs...), pes)
end

function scalar_factors_quotient(bpc::AbstractBeliefPropagationCache)
  return vertex_scalars(bpc), edge_scalars(bpc)
end

function incoming_messages(
  bpc::AbstractBeliefPropagationCache,
  partition_vertices::Vector{<:PartitionVertex};
  ignore_edges=(),
)
  bpes = boundary_partitionedges(bpc, partition_vertices; dir=:in)
  ms = messages(bpc, setdiff(bpes, ignore_edges))
  return reduce(vcat, ms; init=ITensor[])
end

function incoming_messages(
  bpc::AbstractBeliefPropagationCache, partition_vertex::PartitionVertex; kwargs...
)
  return incoming_messages(bpc, [partition_vertex]; kwargs...)
end

#Adapt interface for changing device
function map_messages(f, bpc::AbstractBeliefPropagationCache)
  bpc = copy(bpc)
  for pe in keys(messages(bpc))
    set_message!(bpc, pe, f.(message(bpc, pe)))
  end
  return bpc
end
function map_factors(f, bpc::AbstractBeliefPropagationCache)
  bpc = copy(bpc)
  for v in vertices(bpc)
    @preserve_graph bpc[v] = f(bpc[v])
  end
  return bpc
end
adapt_messages(to, bpc::AbstractBeliefPropagationCache) = map_messages(adapt(to), bpc)
adapt_factors(to, bpc::AbstractBeliefPropagationCache) = map_factors(adapt(to), bpc)

function Adapt.adapt_structure(to, bpc::AbstractBeliefPropagationCache)
  bpc = adapt_messages(to, bpc)
  bpc = adapt_factors(to, bpc)
  return bpc
end

#Forward from partitioned graph
for f in [
  :(PartitionedGraphs.partitioned_graph),
  :(PartitionedGraphs.partitionedge),
  :(PartitionedGraphs.partitionvertices),
  :(PartitionedGraphs.vertices),
  :(PartitionedGraphs.boundary_partitionedges),
]
  @eval begin
    function $f(bpc::AbstractBeliefPropagationCache, args...; kwargs...)
      return $f(partitioned_tensornetwork(bpc), args...; kwargs...)
    end
  end
end

function linkinds(bpc::AbstractBeliefPropagationCache, pe::PartitionEdge)
  return linkinds(partitioned_tensornetwork(bpc), pe)
end

NDTensors.scalartype(bpc::AbstractBeliefPropagationCache) = scalartype(tensornetwork(bpc))

"""
Update the tensornetwork inside the cache out-of-place
"""
function update_factors(bpc::AbstractBeliefPropagationCache, factors)
  bpc = copy(bpc)
  for vertex in eachindex(factors)
    # TODO: Add a check that this preserves the graph structure.
    setindex_preserve_graph!(bpc, factors[vertex], vertex)
  end
  return bpc
end

function update_factor(bpc, vertex, factor)
  bpc = copy(bpc)
  setindex_preserve_graph!(bpc, factor, vertex)
  return bpc
end

function message(bpc::AbstractBeliefPropagationCache, edge::PartitionEdge; kwargs...)
  mts = messages(bpc)
  return get(() -> default_message(bpc, edge; kwargs...), mts, edge)
end
function messages(bpc::AbstractBeliefPropagationCache, edges; kwargs...)
  return map(edge -> message(bpc, edge; kwargs...), edges)
end
function set_messages!(bpc::AbstractBeliefPropagationCache, partitionedges_messages)
  ms = messages(bpc)
  for pe in eachindex(partitionedges_messages)
    # TODO: Add a check that this preserves the graph structure.
    set!(ms, pe, partitionedges_messages[pe])
  end
  return bpc
end
function set_message!(bpc::AbstractBeliefPropagationCache, pe::PartitionEdge, message)
  ms = messages(bpc)
  set!(ms, pe, message)
  return bpc
end

function set_messages(bpc::AbstractBeliefPropagationCache, partitionedges_messages)
  bpc = copy(bpc)
  return set_messages!(bpc, partitionedges_messages)
end
function set_message(bpc::AbstractBeliefPropagationCache, pe::PartitionEdge, message)
  bpc = copy(bpc)
  return set_message!(bpc, pe, message)
end
function delete_messages!(
  bpc::AbstractBeliefPropagationCache, pes::Vector{<:PartitionEdge}=keys(messages(bpc))
)
  ms = messages(bpc)
  for pe in pes
    delete!(ms, pe)
  end
  return bpc
end
function delete_message!(bpc::AbstractBeliefPropagationCache, pe::PartitionEdge)
  return delete_messages!(bpc, [pe])
end
function delete_messages(
  bpc::AbstractBeliefPropagationCache, pes::Vector{<:PartitionEdge}=keys(messages(bpc))
)
  bpc = copy(bpc)
  return delete_messages!(bpc, pes)
end
function delete_message(bpc::AbstractBeliefPropagationCache, pe::PartitionEdge)
  return delete_messages(bpc, [pe])
end

"""
Compute message tensor as product of incoming mts and local state
"""
function updated_message(
  bpc::AbstractBeliefPropagationCache,
  edge::PartitionEdge;
  message_update_alg=default_message_update_alg(bpc),
  kwargs...,
)
  vertex = src(edge)
  incoming_ms = incoming_messages(bpc, vertex; ignore_edges=PartitionEdge[reverse(edge)])
  state = factors(bpc, vertex)

  return message_update(message_update_alg, ITensor[incoming_ms; state]; kwargs...)
end

function update(
  alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache, edge::PartitionEdge; kwargs...
)
  return set_message(bpc, edge, updated_message(bpc, edge; kwargs...))
end

"""
Do a sequential update of the message tensors on `edges`
"""
function update(
  alg::Algorithm,
  bpc::AbstractBeliefPropagationCache,
  edges::Vector;
  (update_diff!)=nothing,
  kwargs...,
)
  bpc = copy(bpc)
  for e in edges
    prev_message = !isnothing(update_diff!) ? message(bpc, e) : nothing
    bpc = update(alg, bpc, e; kwargs...)
    if !isnothing(update_diff!)
      update_diff![] += message_diff(message(bpc, e), prev_message)
    end
  end
  return bpc
end

"""
Do parallel updates between groups of edges of all message tensors
Currently we send the full message tensor data struct to update for each edge_group. But really we only need the
mts relevant to that group.
"""
function update(
  alg::Algorithm,
  bpc::AbstractBeliefPropagationCache,
  edge_groups::Vector{<:Vector{<:PartitionEdge}};
  kwargs...,
)
  new_mts = empty(messages(bpc))
  for edges in edge_groups
    bpc_t = update(alg, bpc, edges; kwargs...)
    for e in edges
      set!(new_mts, e, message(bpc_t, e))
    end
  end
  return set_messages(bpc, new_mts)
end

"""
More generic interface for update, with default params
"""
function update(
  alg::Algorithm,
  bpc::AbstractBeliefPropagationCache;
  edges=alg.kwargs.edge_sequence,
  tol=alg.kwargs.tol,
  maxiter=alg.kwargs.maxiter,
  verbose=alg.kwargs.verbose,
  kwargs...,
)
  compute_error = !isnothing(tol)
  if isnothing(maxiter)
    error("You need to specify a number of iterations for BP!")
  end
  for i in 1:maxiter
    diff = compute_error ? Ref(0.0) : nothing
    bpc = update(alg, bpc, edges; (update_diff!)=diff, kwargs...)
    if compute_error && (diff.x / length(edges)) <= tol
      if verbose
        println("BP converged to desired precision after $i iterations.")
      end
      break
    end
  end
  return bpc
end

function update(bpc::AbstractBeliefPropagationCache; alg=default_update_alg(bpc), kwargs...)
  return update(Algorithm(alg), bpc; kwargs...)
end

function rescale_messages(
  bp_cache::AbstractBeliefPropagationCache, partitionedge::PartitionEdge
)
  return rescale_messages(bp_cache, [partitionedge])
end

function rescale_messages(bp_cache::AbstractBeliefPropagationCache)
  return rescale_messages(bp_cache, partitionedges(bp_cache))
end

function rescale_partitions(
  bpc::AbstractBeliefPropagationCache,
  partitions::Vector;
  verts::Vector=vertices(bpc, partitions),
)
  bpc = copy(bpc)
  tn = tensornetwork(bpc)
  norms = map(v -> inv(norm(tn[v])), verts)
  scale!(bpc, Dictionary(verts, norms))

  vertices_weights = Dictionary()
  for pv in partitions
    pv_vs = filter(v -> v ∈ verts, vertices(bpc, pv))
    isempty(pv_vs) && continue

    vn = region_scalar(bpc, pv)
    s = isreal(vn) ? sign(vn) : 1.0
    vn = s * inv(vn^(1 / length(pv_vs)))
    set!(vertices_weights, first(pv_vs), s*vn)
    for v in pv_vs[2:length(pv_vs)]
      set!(vertices_weights, v, vn)
    end
  end

  scale!(bpc, vertices_weights)

  return bpc
end

function rescale_partitions(bpc::AbstractBeliefPropagationCache, args...; kwargs...)
  return rescale_partitions(bpc, collect(partitions(bpc)), args...; kwargs...)
end

function rescale_partition(
  bpc::AbstractBeliefPropagationCache, partition, args...; kwargs...
)
  return rescale_partitions(bpc, [partition], args...; kwargs...)
end

function rescale(bpc::AbstractBeliefPropagationCache, args...; kwargs...)
  bpc = rescale_messages(bpc)
  bpc = rescale_partitions(bpc, args...; kwargs...)
  return bpc
end

function logscalar(bpc::AbstractBeliefPropagationCache)
  numerator_terms, denominator_terms = scalar_factors_quotient(bpc)
  if any(t -> real(t) < 0, numerator_terms)
    numerator_terms = complex.(numerator_terms)
  end
  if any(t -> real(t) < 0, denominator_terms)
    denominator_terms = complex.(denominator_terms)
  end

  any(iszero, denominator_terms) && return -Inf
  return sum(log.(numerator_terms)) - sum(log.((denominator_terms)))
end

function ITensors.scalar(bpc::AbstractBeliefPropagationCache)
  return exp(logscalar(bpc))
end
