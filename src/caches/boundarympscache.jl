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

struct BoundaryMPSCache{BPC,PG} <: AbstractBeliefPropagationCache
  bp_cache::BPC
  partitionedplanargraph::PG
  maximum_virtual_dimension::Int64
end

bp_cache(bmpsc::BoundaryMPSCache) = bmpsc.bp_cache
partitionedplanargraph(bmpsc::BoundaryMPSCache) = bmpsc.partitionedplanargraph
maximum_virtual_dimension(bmpsc::BoundaryMPSCache) = bmpsc.maximum_virtual_dimension
ppg(bmpsc) = partitionedplanargraph(bmpsc)
planargraph(bmpsc::BoundaryMPSCache) = unpartitioned_graph(partitionedplanargraph(bmpsc))

function partitioned_tensornetwork(bmpsc::BoundaryMPSCache)
  return partitioned_tensornetwork(bp_cache(bmpsc))
end
messages(bmpsc::BoundaryMPSCache) = messages(bp_cache(bmpsc))

default_message_update_alg(bmpsc::BoundaryMPSCache) = "orthogonal"

function default_bp_maxiter(alg::Algorithm"orthogonal", bmpsc::BoundaryMPSCache)
  return default_bp_maxiter(partitioned_graph(ppg(bmpsc)))
end
default_bp_maxiter(alg::Algorithm"biorthogonal", bmpsc::BoundaryMPSCache) = 50
function default_edge_sequence(alg::Algorithm, bmpsc::BoundaryMPSCache)
  return pair.(default_edge_sequence(ppg(bmpsc)))
end
function default_message_update_kwargs(alg::Algorithm"orthogonal", bmpsc::BoundaryMPSCache)
  return (; niters=50, tolerance=1e-10)
end
default_message_update_kwargs(alg::Algorithm"biorthogonal", bmpsc::BoundaryMPSCache) = (;)

function Base.copy(bmpsc::BoundaryMPSCache)
  return BoundaryMPSCache(
    copy(bp_cache(bmpsc)), copy(ppg(bmpsc)), maximum_virtual_dimension(bmpsc)
  )
end

function default_message(bmpsc::BoundaryMPSCache, args...; kwargs...)
  return default_message(bp_cache(bmpsc), args...; kwargs...)
end

function virtual_index_dimension(
  bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge
)
  pes = planargraph_partitionpair_partitionedges(
    bmpsc, planargraph_partitionpair(bmpsc, pe1)
  )
  if findfirst(x -> x == pe1, pes) > findfirst(x -> x == pe2, pes)
    inds_above = reduce(
      vcat, [linkinds(bmpsc, pe) for pe in partitionedges_above(bmpsc, pe2)]
    )
    inds_below = reduce(
      vcat, [linkinds(bmpsc, pe) for pe in partitionedges_below(bmpsc, pe1)]
    )
  else
    inds_above = reduce(
      vcat, [linkinds(bmpsc, pe) for pe in partitionedges_above(bmpsc, pe1)]
    )
    inds_below = reduce(
      vcat, [linkinds(bmpsc, pe) for pe in partitionedges_below(bmpsc, pe2)]
    )
  end
  return minimum((
    prod(dim.(inds_above)), prod(dim.(inds_below)), maximum_virtual_dimension(bmpsc)
  ))
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
  bpc::BeliefPropagationCache;
  grouping_function::Function=v -> first(v),
  message_rank::Int64=1,
)
  bpc = insert_pseudo_planar_edges(bpc; grouping_function)
  planar_graph = partitioned_graph(bpc)
  vertex_groups = group(grouping_function, collect(vertices(planar_graph)))
  ppg = PartitionedGraph(planar_graph, vertex_groups)
  bmpsc = BoundaryMPSCache(bpc, ppg, message_rank)
  return set_interpartition_messages(bmpsc)
end

function BoundaryMPSCache(tn::AbstractITensorNetwork, args...; kwargs...)
  return BoundaryMPSCache(BeliefPropagationCache(tn, args...); kwargs...)
end

#Get all partitionedges within a column/row, ordered top to bottom
function planargraph_partitionedges(bmpsc::BoundaryMPSCache, partition::Int64)
  vs = sort(planargraph_vertices(bmpsc, partition))
  return PartitionEdge.([vs[i] => vs[i + 1] for i in 1:(length(vs) - 1)])
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

#Functions to get the parellel partitionedges sitting above and below a partitionedge
function partitionedges_above(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pes = planargraph_partitionpair_partitionedges(
    bmpsc, planargraph_partitionpair(bmpsc, pe)
  )
  pe_pos = only(findall(x -> x == pe, pes))
  return PartitionEdge[pes[i] for i in (pe_pos + 1):length(pes)]
end

function partitionedges_below(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pes = planargraph_partitionpair_partitionedges(
    bmpsc, planargraph_partitionpair(bmpsc, pe)
  )
  pe_pos = only(findall(x -> x == pe, pes))
  return PartitionEdge[pes[i] for i in 1:(pe_pos - 1)]
end

function partitionedge_above(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pes_above = partitionedges_above(bmpsc, pe)
  isempty(pes_above) && return nothing
  return first(pes_above)
end

function partitionedge_below(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pes_below = partitionedges_below(bmpsc, pe)
  isempty(pes_below) && return nothing
  return last(pes_below)
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

#Initialise all the message tensors for the pairs of neighboring partitions, with virtual rank given by message rank
function set_interpartition_messages(
  bmpsc::BoundaryMPSCache, partitionpairs::Vector{<:Pair}
)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  for partitionpair in partitionpairs
    pes = planargraph_partitionpair_partitionedges(bmpsc, partitionpair)
    for pe in pes
      set!(ms, pe, ITensor[dense(delta(linkinds(bmpsc, pe)))])
    end
    for i in 1:(length(pes) - 1)
      virt_dim = virtual_index_dimension(bmpsc, pes[i], pes[i + 1])
      ind = Index(virt_dim, "m$(i)$(i+1)")
      m1, m2 = only(ms[pes[i]]), only(ms[pes[i + 1]])
      set!(ms, pes[i], ITensor[m1 * delta(ind)])
      set!(ms, pes[i + 1], ITensor[m2 * delta(ind)])
    end
  end
  return bmpsc
end

#Initialise all the interpartition message tensors with virtual rank given by message rank
function set_interpartition_messages(bmpsc::BoundaryMPSCache)
  partitionpairs = pair.(partitionedges(ppg(bmpsc)))
  return set_interpartition_messages(bmpsc, vcat(partitionpairs, reverse.(partitionpairs)))
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
function partition_update(bmpsc::BoundaryMPSCache, partition::Int64)
  vs = sort(planargraph_vertices(bmpsc, partition))
  bmpsc = partition_update(bmpsc, first(vs))
  bmpsc = partition_update(bmpsc, last(vs))
  return bmpsc
end

function partition_update_sequence(bmpsc::BoundaryMPSCache, v1, v2)
  return PartitionEdge.(a_star(ppg(bmpsc), v1, v2))
end
function partition_update_sequence(bmpsc::BoundaryMPSCache, v)
  pv = planargraph_partition(bmpsc, v)
  g = subgraph(unpartitioned_graph(ppg(bmpsc)), planargraph_vertices(bmpsc, pv))
  return PartitionEdge.(post_order_dfs_edges(g, v))
end

#Update all messages within a partition along the path from from v1 to v2
function partition_update(bmpsc::BoundaryMPSCache, args...)
  return update(
    Algorithm("SimpleBP"),
    bmpsc,
    partition_update_sequence(bmpsc, args...);
    message_update_kwargs=(; normalize=false),
  )
end

#Move the orthogonality centre one step on an interpartition from the message tensor on pe1 to that on pe2 
function gauge_step(
  alg::Algorithm"orthogonal",
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
  alg::Algorithm"biorthogonal",
  bmpsc::BoundaryMPSCache,
  pe1::PartitionEdge,
  pe2::PartitionEdge,
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
    bmpsc = gauge_step(alg::Algorithm, bmpsc, pe1, pe2; kwargs...)
  end
  return bmpsc
end

function gauge(alg::Algorithm, bmpsc::BoundaryMPSCache, args...; kwargs...)
  return gauge_walk(alg, bmpsc, mps_gauge_update_sequence(bmpsc, args...); kwargs...)
end

#Move the orthogonality centre on an interpartition to the message tensor on pe or between two pes
function ITensorMPS.orthogonalize(bmpsc::BoundaryMPSCache, args...; kwargs...)
  return gauge(Algorithm("orthogonal"), bmpsc, args...; kwargs...)
end

#Move the biorthogonality centre on an interpartition to the message tensor or between two pes
function biorthogonalize(bmpsc::BoundaryMPSCache, args...; kwargs...)
  return gauge(Algorithm("biorthogonal"), bmpsc, args...; kwargs...)
end

function default_inserter(
  alg::Algorithm"orthogonal",
  bmpsc::BoundaryMPSCache,
  pe::PartitionEdge,
  me::Vector{ITensor},
)
  return set_message(bmpsc, reverse(pe), dag.(me))
end

function default_inserter(
  alg::Algorithm"biorthogonal",
  bmpsc::BoundaryMPSCache,
  pe::PartitionEdge,
  me::Vector{ITensor},
)
  p_above, p_below = partitionedge_above(bmpsc, pe), partitionedge_below(bmpsc, pe)
  me = only(me)
  me_prev = only(message(bmpsc, pe))
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
  return set_message(bmpsc, pe, ITensor[me])
end

function default_updater(
  alg::Algorithm"orthogonal", bmpsc::BoundaryMPSCache, prev_pe, update_pe, prev_v, cur_v
)
  bmpsc = if !isnothing(prev_pe)
    gauge(alg, bmpsc, reverse(prev_pe), reverse(update_pe))
  else
    gauge(alg, bmpsc, reverse(update_pe))
  end
  bmpsc = if !isnothing(prev_v)
    partition_update(bmpsc, prev_v, cur_v)
  else
    partition_update(bmpsc, cur_v)
  end
  return bmpsc
end

function default_updater(
  alg::Algorithm"biorthogonal", bmpsc::BoundaryMPSCache, prev_pe, update_pe, prev_v, cur_v
)
  bmpsc = if !isnothing(prev_pe)
    gauge(alg, bmpsc, prev_pe, update_pe)
  else
    gauge(alg, bmpsc, update_pe)
  end
  bmpsc = if !isnothing(prev_v)
    partition_update(bmpsc, prev_v, cur_v)
  else
    partition_update(bmpsc, cur_v)
  end
  return bmpsc
end

function default_cache_prep_function(
  alg::Algorithm"biorthogonal", bmpsc::BoundaryMPSCache, partitionpair
)
  return bmpsc
end
function default_cache_prep_function(
  alg::Algorithm"orthogonal", bmpsc::BoundaryMPSCache, partitionpair
)
  return switch_messages(bmpsc, partitionpair)
end

default_niters(alg::Algorithm"orthogonal") = 25
default_niters(alg::Algorithm"biorthogonal") = 3
default_tolerance(alg::Algorithm"orthogonal") = 1e-10
default_tolerance(alg::Algorithm"biorthogonal") = nothing

function default_costfunction(
  alg::Algorithm"orthogonal",
  bmpsc::BoundaryMPSCache,
  pe::PartitionEdge,
  me::Vector{ITensor},
)
  return region_scalar(bp_cache(bmpsc), src(pe)) / norm(only(me))
end

function default_costfunction(
  alg::Algorithm"biorthogonal",
  bmpsc::BoundaryMPSCache,
  pe::PartitionEdge,
  me::Vector{ITensor},
)
  return region_scalar(bp_cache(bmpsc), src(pe)) /
         dot(only(me), only(message(bmpsc, reverse(pe))))
end

#Update all the message tensors on an interpartition via a specified fitting procedure 
#TODO: Make two-site possible
function update(
  alg::Algorithm,
  bmpsc::BoundaryMPSCache,
  partitionpair::Pair;
  inserter=default_inserter,
  costfunction=default_costfunction,
  updater=default_updater,
  cache_prep_function=default_cache_prep_function,
  niters::Int64=default_niters(alg),
  tolerance=default_tolerance(alg),
  normalize=true,
)
  bmpsc = cache_prep_function(alg, bmpsc, partitionpair)
  pes = planargraph_partitionpair_partitionedges(bmpsc, partitionpair)
  update_seq = vcat(pes, reverse(pes)[2:length(pes)])
  prev_v, prev_pe = nothing, nothing
  prev_cf = 0
  for i in 1:niters
    cf = 0
    for update_pe in update_seq
      cur_v = parent(src(update_pe))
      bmpsc = updater(alg, bmpsc, prev_pe, update_pe, prev_v, cur_v)
      me = updated_message(
        bmpsc, update_pe; message_update=ms -> default_message_update(ms; normalize)
      )
      cf += costfunction(alg, bmpsc, update_pe, me)
      bmpsc = inserter(alg, bmpsc, update_pe, me)
      prev_v, prev_pe = cur_v, update_pe
    end
    epsilon = abs(cf - prev_cf) / length(update_seq)
    if !isnothing(tolerance) && epsilon < tolerance
      return cache_prep_function(alg, bmpsc, partitionpair)
    else
      prev_cf = cf
    end
  end
  return cache_prep_function(alg, bmpsc, partitionpair)
end

#Assume all vertices live in the same partition for now
function ITensorNetworks.environment(bmpsc::BoundaryMPSCache, verts::Vector; kwargs...)
  vs = parent.((partitionvertices(bp_cache(bmpsc), verts)))
  pv = only(planargraph_partitions(bmpsc, vs))
  bmpsc = partition_update(bmpsc, pv)
  return environment(bp_cache(bmpsc), verts; kwargs...)
end
