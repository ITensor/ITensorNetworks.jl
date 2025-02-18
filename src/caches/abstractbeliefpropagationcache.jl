using Graphs: IsDirected
using SplitApplyCombine: group
using LinearAlgebra: diag, dot
using ITensors: dir
using ITensorMPS: ITensorMPS
using NamedGraphs.PartitionedGraphs:
  PartitionedGraphs,
  PartitionedGraph,
  PartitionVertex,
  boundary_partitionedges,
  partitionvertices,
  partitionedges,
  unpartitioned_graph
using SimpleTraits: SimpleTraits, Not, @traitfn
using NDTensors: NDTensors

abstract type AbstractBeliefPropagationCache end

function default_message_update(contract_list::Vector{ITensor}; normalize=true, kwargs...)
  sequence = optimal_contraction_sequence(contract_list)
  updated_messages = contract(contract_list; sequence, kwargs...)
  message_norm = norm(updated_messages)
  if normalize && !iszero(message_norm)
    updated_messages /= message_norm
  end
  return ITensor[updated_messages]
end

#TODO: Take `dot` without precontracting the messages to allow scaling to more complex messages
function message_diff(message_a::Vector{ITensor}, message_b::Vector{ITensor})
  lhs, rhs = contract(message_a), contract(message_b)
  f = abs2(dot(lhs / norm(lhs), rhs / norm(rhs)))
  return 1 - f
end

default_message(elt, inds_e) = ITensor[denseblocks(delta(elt, i)) for i in inds_e]
default_messages(ptn::PartitionedGraph) = Dictionary()
@traitfn default_bp_maxiter(g::::(!IsDirected)) = is_tree(g) ? 1 : nothing
@traitfn function default_bp_maxiter(g::::IsDirected)
  return default_bp_maxiter(undirected_graph(underlying_graph(g)))
end
default_partitioned_vertices(ψ::AbstractITensorNetwork) = group(v -> v, vertices(ψ))
function default_partitioned_vertices(f::AbstractFormNetwork)
  return group(v -> original_state_vertex(f, v), vertices(f))
end

partitioned_tensornetwork(bpc::AbstractBeliefPropagationCache) = not_implemented()
messages(bpc::AbstractBeliefPropagationCache) = not_implemented()
function default_message(
  bpc::AbstractBeliefPropagationCache, edge::PartitionEdge; kwargs...
)
  return not_implemented()
end
default_message_update_alg(bpc::AbstractBeliefPropagationCache) = not_implemented()
Base.copy(bpc::AbstractBeliefPropagationCache) = not_implemented()
default_bp_maxiter(alg::Algorithm, bpc::AbstractBeliefPropagationCache) = not_implemented()
function default_edge_sequence(alg::Algorithm, bpc::AbstractBeliefPropagationCache)
  return not_implemented()
end
function default_message_update_kwargs(alg::Algorithm, bpc::AbstractBeliefPropagationCache)
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
partitionpairs(bpc::AbstractBeliefPropagationCache) = not_implemented()

function default_edge_sequence(
  bpc::AbstractBeliefPropagationCache; alg=default_message_update_alg(bpc)
)
  return default_edge_sequence(Algorithm(alg), bpc)
end
function default_bp_maxiter(
  bpc::AbstractBeliefPropagationCache; alg=default_message_update_alg(bpc)
)
  return default_bp_maxiter(Algorithm(alg), bpc)
end
function default_message_update_kwargs(
  bpc::AbstractBeliefPropagationCache; alg=default_message_update_alg(bpc)
)
  return default_message_update_kwargs(Algorithm(alg), bpc)
end

function tensornetwork(bpc::AbstractBeliefPropagationCache)
  return unpartitioned_graph(partitioned_tensornetwork(bpc))
end

function factors(bpc::AbstractBeliefPropagationCache, verts::Vector)
  return ITensor[tensornetwork(bpc)[v] for v in verts]
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
  bpc::AbstractBeliefPropagationCache, pes=partitionpairs(bpc); kwargs...
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

#Forward from partitioned graph
for f in [
  :(PartitionedGraphs.partitioned_graph),
  :(PartitionedGraphs.partitionedge),
  :(PartitionedGraphs.partitionvertices),
  :(PartitionedGraphs.vertices),
  :(PartitionedGraphs.boundary_partitionedges),
  :(ITensorMPS.linkinds),
]
  @eval begin
    function $f(bpc::AbstractBeliefPropagationCache, args...; kwargs...)
      return $f(partitioned_tensornetwork(bpc), args...; kwargs...)
    end
  end
end

NDTensors.scalartype(bpc::AbstractBeliefPropagationCache) = scalartype(tensornetwork(bpc))

"""
Update the tensornetwork inside the cache
"""
function update_factors(bpc::AbstractBeliefPropagationCache, factors)
  bpc = copy(bpc)
  tn = tensornetwork(bpc)
  for vertex in eachindex(factors)
    # TODO: Add a check that this preserves the graph structure.
    setindex_preserve_graph!(tn, factors[vertex], vertex)
  end
  return bpc
end

function update_factor(bpc, vertex, factor)
  return update_factors(bpc, Dictionary([vertex], [factor]))
end

function message(bpc::AbstractBeliefPropagationCache, edge::PartitionEdge; kwargs...)
  mts = messages(bpc)
  return get(() -> default_message(bpc, edge; kwargs...), mts, edge)
end
function messages(bpc::AbstractBeliefPropagationCache, edges; kwargs...)
  return map(edge -> message(bpc, edge; kwargs...), edges)
end
function set_message(bpc::AbstractBeliefPropagationCache, pe::PartitionEdge, message)
  bpc = copy(bpc)
  ms = messages(bpc)
  set!(ms, pe, message)
  return bpc
end

"""
Compute message tensor as product of incoming mts and local state
"""
function updated_message(
  bpc::AbstractBeliefPropagationCache,
  edge::PartitionEdge;
  message_update_function=default_message_update,
  message_update_function_kwargs=(;),
)
  vertex = src(edge)
  incoming_ms = incoming_messages(bpc, vertex; ignore_edges=PartitionEdge[reverse(edge)])
  state = factors(bpc, vertex)

  return message_update_function(
    ITensor[incoming_ms; state]; message_update_function_kwargs...
  )
end

function update(
  alg::Algorithm"simplebp",
  bpc::AbstractBeliefPropagationCache,
  edge::PartitionEdge;
  kwargs...,
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
  new_mts = copy(messages(bpc))
  for edges in edge_groups
    bpc_t = update(alg, bpc, edges; kwargs...)
    for e in edges
      new_mts[e] = message(bpc_t, e)
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
  edges=default_edge_sequence(alg, bpc),
  maxiter=default_bp_maxiter(alg, bpc),
  message_update_kwargs=default_message_update_kwargs(alg, bpc),
  tol=nothing,
  verbose=false,
)
  compute_error = !isnothing(tol)
  if isnothing(maxiter)
    error("You need to specify a number of iterations for BP!")
  end
  for i in 1:maxiter
    diff = compute_error ? Ref(0.0) : nothing
    bpc = update(alg, bpc, edges; (update_diff!)=diff, message_update_kwargs...)
    if compute_error && (diff.x / length(edges)) <= tol
      if verbose
        println("BP converged to desired precision after $i iterations.")
      end
      break
    end
  end
  return bpc
end

function update(
  bpc::AbstractBeliefPropagationCache;
  alg::String=default_message_update_alg(bpc),
  kwargs...,
)
  return update(Algorithm(alg), bpc; kwargs...)
end
