using NamedGraphs: NamedGraphs
using NamedGraphs.GraphsExtensions: add_edges
using ITensorNetworks: ITensorNetworks, BeliefPropagationCache, region_scalar
using ITensorNetworks.ITensorsExtensions: map_diag
using ITensorMPS: ITensorMPS, orthogonalize
using NamedGraphs.PartitionedGraphs:
  partitioned_graph, PartitionVertex, partitionvertex, partitioned_vertices, which_partition
using SplitApplyCombine: group
using ITensors: commoninds, random_itensor
using LinearAlgebra: pinv

struct BoundaryMPSCache{BPC,PG}
  bp_cache::BPC
  partitionedplanargraph::PG
end

bp_cache(bmpsc::BoundaryMPSCache) = bmpsc.bp_cache
partitionedplanargraph(bmpsc::BoundaryMPSCache) = bmpsc.partitionedplanargraph
ppg(bmpsc) = partitionedplanargraph(bmpsc)
planargraph(bmpsc::BoundaryMPSCache) = unpartitioned_graph(partitionedplanargraph(bmpsc))
tensornetwork(bmpsc::BoundaryMPSCache) = tensornetwork(bp_cache(bmpsc))
function partitioned_tensornetwork(bmpsc::BoundaryMPSCache)
  return partitioned_tensornetwork(bp_cache(bmpsc))
end

default_edge_sequence(bmpsc::BoundaryMPSCache) = pair.(default_edge_sequence(ppg(bmpsc)))
function default_bp_maxiter(alg::Algorithm"orthogonal", bmpsc::BoundaryMPSCache)
  return default_bp_maxiter(partitioned_graph(ppg(bmpsc)))
end
default_bp_maxiter(alg::Algorithm"biorthogonal", bmpsc::BoundaryMPSCache) = 50
function default_mps_fit_kwargs(alg::Algorithm"orthogonal", bmpsc::BoundaryMPSCache)
  return (; niters=50, tolerance=1e-10)
end
default_mps_fit_kwargs(alg::Algorithm"biorthogonal", bmpsc::BoundaryMPSCache) = (;)

function Base.copy(bmpsc::BoundaryMPSCache)
  return BoundaryMPSCache(copy(bp_cache(bmpsc)), copy(ppg(bmpsc)))
end

function planargraph_vertices(bmpsc::BoundaryMPSCache, partition::Int64)
  return vertices(ppg(bmpsc), PartitionVertex(partition))
end
function planargraph_partition(bmpsc::BoundaryMPSCache, vertex)
  return parent(partitionvertex(ppg(bmpsc), vertex))
end
function planargraph_partitions(bmpsc::BoundaryMPSCache, verts)
  return parent.(partitionvertices(ppg(bmpsc), verts))
end
function planargraph_partitionpair(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  return pair(partitionedge(ppg(bmpsc), parent(pe)))
end

function BoundaryMPSCache(
  bpc::BeliefPropagationCache; sort_f::Function=v -> first(v), message_rank::Int64=1
)
  bpc = insert_pseudo_planar_edges(bpc; sort_f)
  planar_graph = partitioned_graph(bpc)
  vertex_groups = group(sort_f, collect(vertices(planar_graph)))
  ppg = PartitionedGraph(planar_graph, vertex_groups)
  bmpsc = BoundaryMPSCache(bpc, ppg)
  return set_interpartition_messages(bmpsc, message_rank)
end

function BoundaryMPSCache(tn::AbstractITensorNetwork, args...; kwargs...)
  return BoundaryMPSCache(BeliefPropagationCache(tn, args...); kwargs...)
end

#Get all partitionedges within a column/row, ordered top to bottom
function planargraph_partitionedges(bmpsc::BoundaryMPSCache, partition::Int64)
  vs = sort(planargraph_vertices(bmpsc, partition))
  return PartitionEdge.([vs[i] => vs[i + 1] for i in 1:(length(vs) - 1)])
end

#Functions to get the partitionedge sitting parallel above and below a message tensor
function partitionedge_above(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pes = planargraph_partitionpair_partitionedges(
    bmpsc, planargraph_partitionpair(bmpsc, pe)
  )
  pe_pos = only(findall(x -> x == pe, pes))
  pe_pos == length(pes) && return nothing
  return pes[pe_pos + 1]
end

function partitionedge_below(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pes = planargraph_partitionpair_partitionedges(
    bmpsc, planargraph_partitionpair(bmpsc, pe)
  )
  pe_pos = only(findall(x -> x == pe, pes))
  pe_pos == 1 && return nothing
  return pes[pe_pos - 1]
end

#Get the sequence of pairs partitionedges that need to be updated to move the MPS gauge from pe1 to pe2
function mps_gauge_update_sequence(
  bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge
)
  ppgpe1, ppgpe2 = planargraph_partitionpair(bmpsc, pe1),
  planargraph_partitionpair(bmpsc, pe2)
  @assert ppgpe1 == ppgpe2
  pes = planargraph_partitionpair_partitionedges(bmpsc, ppgpe1)
  return pair_sequence(pes, pe1, pe2)
end

#Get the sequence of pairs partitionedges that need to be updated to move the MPS gauge onto pe
function mps_gauge_update_sequence(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  ppgpe = planargraph_partitionpair(bmpsc, pe)
  pes = planargraph_partitionpair_partitionedges(bmpsc, ppgpe)
  return vcat(
    mps_gauge_update_sequence(bmpsc, last(pes), pe),
    mps_gauge_update_sequence(bmpsc, first(pes), pe),
  )
end

#Get all partitionedges between the pair of neighboring partitions, sorted top to bottom
#TODO: Bring in line with NamedGraphs change
function planargraph_partitionpair_partitionedges(
  bmpsc::BoundaryMPSCache, partitionpair::Pair
)
  pg = planargraph(bmpsc)
  src_vs, dst_vs = planargraph_vertices(bmpsc, first(partitionpair)),
  planargraph_vertices(bmpsc, last(partitionpair))
  es = filter(
    x -> !isempty(last(x)),
    [src_v => intersect(neighbors(pg, src_v), dst_vs) for src_v in src_vs],
  )
  es = map(x -> first(x) => only(last(x)), es)
  return sort(PartitionEdge.(NamedEdge.(es)); by=x -> src(parent(x)))
end

function set_message(bmpsc::BoundaryMPSCache, pe::PartitionEdge, m::Vector{ITensor})
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  set!(ms, pe, m)
  return bmpsc
end

#Initialise all the message tensors for the pair of neighboring partitions, with virtual rank given by message rank
function set_interpartition_messages(
  bmpsc::BoundaryMPSCache, partitionpair::Pair, message_rank::Int64
)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  pes = planargraph_partitionpair_partitionedges(bmpsc, partitionpair)
  prev_virtual_ind = nothing
  for (i, pg_pe) in enumerate(pes)
    siteinds = linkinds(bmpsc, pg_pe)
    next_virtual_index = i != length(pes) ? Index(message_rank, "m$(i)$(i+1)") : nothing
    me = denseblocks(delta(siteinds))
    virt_inds = filter(x -> !isnothing(x), [prev_virtual_ind, next_virtual_index])
    if !isempty(virt_inds)
      me *= delta(virt_inds)
    end
    set!(ms, pg_pe, ITensor[me])
    prev_virtual_ind = next_virtual_index
  end

  return bmpsc
end

#Initialise all the interpartition message tensors with virtual rank given by message rank
function set_interpartition_messages(bmpsc::BoundaryMPSCache, message_rank::Int64=1)
  bmpsc = copy(bmpsc)
  pes = partitionedges(ppg(bmpsc))
  for pe in vcat(pes, reverse(reverse.(pes)))
    bmpsc = set_interpartition_messages(
      bmpsc, parent(src(pe)) => parent(dst(pe)), message_rank
    )
  end
  return bmpsc
end

#Switch the message on partition edge pe with its reverse (and dagger them)
function switch_message(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  me, mer = message(bmpsc, pe), message(bmpsc, reverse(pe))
  set!(ms, pe, dag.(mer))
  set!(ms, reverse(pe), dag.(me))
  return bmpsc
end

#Switch the message tensors from partitionpair  i -> i + 1 with those from i + 1 -> i
function switch_messages(bmpsc::BoundaryMPSCache, partitionpair::Pair)
  for pe in planargraph_partitionpair_partitionedges(bmpsc, partitionpair)
    bmpsc = switch_message(bmpsc, pe)
  end
  return bmpsc
end

#Update all messages tensors within a partition
function partition_update(bmpsc::BoundaryMPSCache, partition::Int64; kwargs...)
  vs = sort(planargraph_vertices(bmpsc, partition))
  bmpsc = partition_update(bmpsc, first(vs); kwargs...)
  bmpsc = partition_update(bmpsc, last(vs); kwargs...)
  return bmpsc
end

#Update all messages within a partition along the path from from v1 to v2
function partition_update(bmpsc::BoundaryMPSCache, v1, v2; kwargs...)
  return update(bmpsc, PartitionEdge.(a_star(ppg(bmpsc), v1, v2)); kwargs...)
end

#Update all message tensors within a partition pointing towards v
function partition_update(bmpsc::BoundaryMPSCache, v; kwargs...)
  pv = planargraph_partition(bmpsc, v)
  g = subgraph(unpartitioned_graph(ppg(bmpsc)), planargraph_vertices(bmpsc, pv))
  return update(bmpsc, PartitionEdge.(post_order_dfs_edges(g, v)); kwargs...)
end

#Move the orthogonality centre one step on an interpartition from the message tensor on pe1 to that on pe2 
function gauge_step(
  alg::Algorithm"orthogonalize",
  bmpsc::BoundaryMPSCache,
  pe1::PartitionEdge,
  pe2::PartitionEdge;
  kwargs...,
)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  m1, m2 = only(message(bmpsc, pe1)), only(message(bmpsc, pe2))
  @assert !isempty(commoninds(m1, m2))
  left_inds = uniqueinds(m1, m2)
  m1, Y = factorize(m1, left_inds; ortho="left", kwargs...)
  m2 = m2 * Y
  set!(ms, pe1, ITensor[m1])
  set!(ms, pe2, ITensor[m2])
  return bmpsc
end

#Move the biorthogonality centre one step on an interpartition from the partition edge pe1 (and its reverse) to that on pe2 
function gauge_step(
  alg::Algorithm"biorthogonalize",
  bmpsc::BoundaryMPSCache,
  pe1::PartitionEdge,
  pe2::PartitionEdge;
  regularization=1e-12,
)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)

  m1, m1r = only(message(bmpsc, pe1)), only(message(bmpsc, reverse(pe1)))
  m2, m2r = only(message(bmpsc, pe2)), only(message(bmpsc, reverse(pe2)))
  top_cind, bottom_cind = commonind(m1, m2), commonind(m1r, m2r)
  m1_siteinds, m2_siteinds = commoninds(m1, m1r), commoninds(m2, m2r)
  top_ncind, bottom_ncind = setdiff(inds(m1), [m1_siteinds; top_cind]),
  setdiff(inds(m1r), [m1_siteinds; bottom_cind])

  E = if isempty(top_ncind)
    m1 * m1r
  else
    m1 * replaceind(m1r, only(bottom_ncind), only(top_ncind))
  end
  U, S, V = svd(E, bottom_cind; alg="recursive")

  S_sqrtinv = map_diag(x -> pinv(sqrt(x)), S)
  S_sqrt = map_diag(x -> sqrt(x), S)

  set!(ms, pe1, ITensor[replaceind((m1 * dag(V)) * S_sqrtinv, commonind(U, S), top_cind)])
  set!(
    ms,
    reverse(pe1),
    ITensor[replaceind((m1r * dag(U)) * S_sqrtinv, commonind(V, S), bottom_cind)],
  )
  set!(ms, pe2, ITensor[replaceind((m2 * V) * S_sqrt, commonind(U, S), top_cind)])
  set!(
    ms, reverse(pe2), ITensor[replaceind((m2r * U) * S_sqrt, commonind(V, S), bottom_cind)]
  )

  return bmpsc
end

#Move the orthogonality / biorthogonality centre on an interpartition via a sequence of steps between message tensors
function gauge_walk(alg::Algorithm, bmpsc::BoundaryMPSCache, seq::Vector; kwargs...)
  for (pe1, pe2) in seq
    bmpsc = gauge_step(alg::Algorithm, bmpsc, pe1, pe2)
  end
  return bmpsc
end

#Move the orthogonality centre on an interpartition to the message tensor on pe or between two pes
function ITensorMPS.orthogonalize(bmpsc::BoundaryMPSCache, args...; kwargs...)
  return gauge_walk(
    Algorithm("orthogonalize"), bmpsc, mps_gauge_update_sequence(bmpsc, args...); kwargs...
  )
end

#Move the biorthogonality centre on an interpartition to the message tensor or between two pes
function biorthogonalize(bmpsc::BoundaryMPSCache, args...; kwargs...)
  return gauge_walk(
    Algorithm("biorthogonalize"),
    bmpsc,
    mps_gauge_update_sequence(bmpsc, args...);
    kwargs...,
  )
end

#Update all the message tensors on an interpartition via an orthogonal fitting procedure 
function mps_update(
  alg::Algorithm"orthogonal",
  bmpsc::BoundaryMPSCache,
  partitionpair::Pair;
  niters::Int64=25,
  tolerance=1e-10,
  normalize=true,
)
  bmpsc = switch_messages(bmpsc, partitionpair)
  pes = planargraph_partitionpair_partitionedges(bmpsc, partitionpair)
  update_seq = vcat(pes, reverse(pes)[2:length(pes)])
  prev_v, prev_pe = nothing, nothing
  prev_costfunction = 0
  for i in 1:niters
    costfunction = 0
    for update_pe in update_seq
      cur_v = parent(src(update_pe))
      bmpsc = if !isnothing(prev_pe)
        orthogonalize(bmpsc, reverse(prev_pe), reverse(update_pe))
      else
        orthogonalize(bmpsc, reverse(update_pe))
      end
      bmpsc = if !isnothing(prev_v)
        partition_update(
        bmpsc,
        prev_v,
        cur_v;
        message_update=ms -> default_message_update(ms; normalize=false),
      )
      else
        partition_update(
        bmpsc, cur_v; message_update=ms -> default_message_update(ms; normalize=false)
      )
      end
      me = update_message(
        bmpsc, update_pe; message_update=ms -> default_message_update(ms; normalize)
      )
      costfunction += region_scalar(bp_cache(bmpsc), src(update_pe)) / norm(me)
      bmpsc = set_message(bmpsc, reverse(update_pe), dag.(me))
      prev_v, prev_pe = cur_v, update_pe
    end
    epsilon = abs(costfunction - prev_costfunction) / length(update_seq)
    if !isnothing(tolerance) && epsilon < tolerance
      return switch_messages(bmpsc, partitionpair)
    else
      prev_costfunction = costfunction
    end
  end
  return switch_messages(bmpsc, partitionpair)
end

#Update all the message tensors on an interpartition via a biorthogonal fitting procedure
function mps_update(
  alg::Algorithm"biorthogonal", bmpsc::BoundaryMPSCache, partitionpair::Pair; normalize=true
)
  prev_v, prev_pe = nothing, nothing
  for update_pe in planargraph_partitionpair_partitionedges(bmpsc, partitionpair)
    cur_v = parent(src(update_pe))
    bmpsc = if !isnothing(prev_pe)
      biorthogonalize(bmpsc, prev_pe, update_pe)
    else
      biorthogonalize(bmpsc, update_pe)
    end
    bmpsc = if !isnothing(prev_v)
      partition_update(
      bmpsc,
      prev_v,
      cur_v;
      message_update=ms -> default_message_update(ms; normalize=false),
    )
    else
      partition_update(
      bmpsc, cur_v; message_update=ms -> default_message_update(ms; normalize=false)
    )
    end

    me_prev = only(message(bmpsc, update_pe))
    me = only(
      update_message(
        bmpsc, update_pe; message_update=ms -> default_message_update(ms; normalize)
      ),
    )
    p_above, p_below = partitionedge_above(bmpsc, update_pe),
    partitionedge_below(bmpsc, update_pe)
    if !isnothing(p_above)
      me = replaceind(
        me,
        commonind(me, only(message(bmpsc, reverse(p_above)))),
        commonind(me_prev, only(message(bmpsc, p_above))),
      )
    end
    if !isnothing(p_below)
      me = replaceind(
        me,
        commonind(me, only(message(bmpsc, reverse(p_below)))),
        commonind(me_prev, only(message(bmpsc, p_below))),
      )
    end
    bmpsc = set_message(bmpsc, update_pe, ITensor[me])
    prev_v, prev_pe = cur_v, update_pe
  end

  return bmpsc
end

"""
More generic interface for update, with default params
"""
function update(
  alg::Algorithm,
  bmpsc::BoundaryMPSCache;
  partitionpairs=default_edge_sequence(bmpsc),
  maxiter=default_bp_maxiter(alg, bmpsc),
  mps_fit_kwargs=default_mps_fit_kwargs(alg, bmpsc),
)
  if isnothing(maxiter)
    error("You need to specify a number of iterations for Boundary MPS!")
  end
  for i in 1:maxiter
    for partitionpair in partitionpairs
      bmpsc = mps_update(alg, bmpsc, partitionpair; mps_fit_kwargs...)
    end
  end
  return bmpsc
end

function update(bmpsc::BoundaryMPSCache; alg::String="orthogonal", kwargs...)
  return update(Algorithm(alg), bmpsc; kwargs...)
end

#Assume all vertices live in the same partition for now
function ITensorNetworks.environment(bmpsc::BoundaryMPSCache, verts::Vector; kwargs...)
  vs = parent.((partitionvertices(bp_cache(bmpsc), verts)))
  pv = only(planargraph_partitions(bmpsc, vs))
  bmpsc = partition_update(bmpsc, pv)
  return environment(bp_cache(bmpsc), verts; kwargs...)
end

function ITensorNetworks.environment(bmpsc::BoundaryMPSCache, vertex; kwargs...)
  return environment(bmpsc, [vertex]; kwargs...)
end

#Forward onto beliefpropagationcache
for f in [
  :messages,
  :message,
  :update_message,
  :(ITensorNetworks.linkinds),
  :default_edge_sequence,
  :factor,
  :factors,
]
  @eval begin
    function $f(bmpsc::BoundaryMPSCache, args...; kwargs...)
      return $f(bp_cache(bmpsc), args...; kwargs...)
    end
  end
end

#Wrap around beliefpropagationcache
for f in [:update_factors, :update_factor]
  @eval begin
    function $f(bmpsc::BoundaryMPSCache, args...; kwargs...)
      bmpsc = copy(bmpsc)
      bpc = bp_cache(bmpsc)
      bpc = $f(bpc, args...; kwargs...)
      return BoundaryMPSCache(bpc, partitionedplanargraph(bmpsc))
    end
  end
end

#Wrap around beliefpropagationcache but only for specific argument
function update(bmpsc::BoundaryMPSCache, pes::Vector{<:PartitionEdge}; kwargs...)
  bmpsc = copy(bmpsc)
  bpc = bp_cache(bmpsc)
  bpc = update(bpc, pes; kwargs...)
  return BoundaryMPSCache(bpc, partitionedplanargraph(bmpsc))
end
