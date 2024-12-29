using NamedGraphs: NamedGraphs
using NamedGraphs.GraphsExtensions: add_edges
using ITensorNetworks: ITensorNetworks, BeliefPropagationCache, region_scalar
using NamedGraphs.PartitionedGraphs: partitioned_graph, PartitionVertex, partitionvertex, partitioned_vertices,
  which_partition
using SplitApplyCombine: group
using ITensors: commoninds

struct BoundaryMPSCache{BPC,PG}
    bp_cache::BPC
    partitionedplanargraph::PG
end

bp_cache(bmpsc::BoundaryMPSCache) = bmpsc.bp_cache
partitionedplanargraph(bmpsc::BoundaryMPSCache) = bmpsc.partitionedplanargraph
ppg(bmpsc) = partitionedplanargraph(bmpsc)
planargraph(bmpsc::BoundaryMPSCache) = unpartitioned_graph(partitionedplanargraph(bmpsc))

function Base.copy(bmpsc::BoundaryMPSCache)
  return BoundaryMPSCache(copy(bp_cache(bmpsc)), copy(ppg(bmpsc)))
end

planargraph_vertices(bmpsc::BoundaryMPSCache, pv::PartitionVertex) = vertices(ppg(bmpsc), pv)
planargraph_partitionvertex(bmpsc::BoundaryMPSCache, vertex) = partitionvertex(ppg(bmpsc), vertex)
planargraph_partitionvertices(bmpsc::BoundaryMPSCache, verts) = partitionvertices(ppg(bmpsc), verts)
planargraph_partitionedge(bmpsc::BoundaryMPSCache, pe::PartitionEdge) = partitionedge(ppg(bmpsc), parent(pe))

function BoundaryMPSCache(bpc::BeliefPropagationCache; sort_f::Function = v -> first(v),
  message_rank::Int64=1)
  bpc = insert_missing_planar_edges(bpc; sort_f)
  planar_graph = partitioned_graph(bpc)
  vertex_groups = group(sort_f, collect(vertices(planar_graph)))
  ppg = PartitionedGraph(planar_graph, vertex_groups)
  bmpsc = BoundaryMPSCache(bpc, ppg)
  return initialize_messages(bmpsc, message_rank)
end

#Get all partitionedges within a column/row partition, sorted top to bottom 
function planargraph_partitionedges(bmpsc::BoundaryMPSCache, pv::PartitionVertex)
  vs = sort(planargraph_vertices(bmpsc, pv))
  return PartitionEdge.([vs[i] => vs[i+1] for i in 1:(length(vs)-1)])
end

#Sequence of Edges from pe1 to pe2 along a column/row interpartition
function update_sequence(bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge)
  ppgpe1, ppgpe2 = planargraph_partitionedge(bmpsc, pe1), planargraph_partitionedge(bmpsc, pe2)
  @assert ppgpe1 == ppgpe2
  pes = planargraph_partitionedges(bmpsc, ppgpe1)
  pe1_pos, pe2_pos = only(findall(x->x==pe1, pes)), only(findall(x->x==pe2, pes))
  pe1_pos < pe2_pos && return PartitionEdge.([parent(pes[i]) => parent(pes[i+1]) for i in pe1_pos:(pe2_pos-1)])
  return PartitionEdge.([parent(pes[i]) => parent(pes[i-1]) for i in pe1_pos:-1:(pe2_pos+1)])
end

#Edges toward pe1 along a column/row interpartition
function update_sequence(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  ppgpe = planargraph_partitionedge(bmpsc, pe)
  pes =  planargraph_partitionedges(bmpsc, ppgpe)
  return vcat(update_sequence(bmpsc, last(pes), pe), update_sequence(bmpsc, first(pes), pe))
end

#Get all partitionedges from src(pe) to dst(pe), sorted top to bottom
#TODO: Bring in line with NamedGraphs change
function planargraph_partitionedges(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pg = planargraph(bmpsc)
  src_vs, dst_vs = planargraph_vertices(bmpsc, src(pe)), planargraph_vertices(bmpsc, dst(pe))
  es = filter(x -> !isempty(last(x)), [src_v => intersect(neighbors(pg, src_v), dst_vs) for src_v in src_vs])
  es = map(x -> first(x) => only(last(x)), es)
  return sort(PartitionEdge.(NamedEdge.(es)); by = x -> src(parent(x)))
end

function set_message(bmpsc::BoundaryMPSCache, pe::PartitionEdge, m::Vector{ITensor})
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  set!(ms, pe, m)
  return bmpsc
end

function initialize_messages(bmpsc::BoundaryMPSCache, pe::PartitionEdge, message_rank::Int64)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  pes = planargraph_partitionedges(bmpsc, pe)
  prev_virtual_ind = nothing
  for (i, pg_pe) in enumerate(pes)
    siteinds = linkinds(bmpsc, pg_pe)
    next_virtual_index = i != length(pes) ? Index(message_rank, "m$(i)$(i+1)") : nothing
    inds = filter(x -> !isnothing(x), [siteinds; prev_virtual_ind; next_virtual_index])
    set!(ms, pg_pe, ITensor[delta(inds)])
    prev_virtual_ind = next_virtual_index
  end
  return bmpsc
end

function initialize_messages(bmpsc::BoundaryMPSCache, message_rank::Int64 = 1)
  bmpsc = copy(bmpsc)
  pes = partitionedges(ppg(bmpsc))
  for pe in vcat(pes, reverse(reverse.(pes)))
    bmpsc = initialize_messages(bmpsc, pe, message_rank)
  end
  return bmpsc
end

#Switch the messages from column/row  i -> i + 1 with those from i + 1 -> i
function switch_messages(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  pes = planargraph_partitionedges(bmpsc, pe)
  for pe_i in pes
    me, mer = message(bmpsc, pe_i), message(bmpsc, reverse(pe_i))
    set!(ms, pe_i, dag.(mer))
    set!(ms, reverse(pe_i), dag.(me))
  end
  return bmpsc
end

#Update all messages flowing within a partition
function partition_update(bmpsc::BoundaryMPSCache, pv::PartitionVertex; kwargs...)
  vs = sort(planargraph_vertices(bmpsc, pv))
  bmpsc = partition_update(bmpsc, first(vs); kwargs...)
  bmpsc = partition_update(bmpsc, last(vs); kwargs...)
  return bmpsc
end

#Update all messages flowing within a partition from v1 to v2
function partition_update(bmpsc::BoundaryMPSCache, v1, v2; kwargs...)
  return update(bmpsc, PartitionEdge.(a_star(ppg(bmpsc), v1, v2)); kwargs...)
end

#Update all messages flowing within a partition from v1 to v2
function partition_update(bmpsc::BoundaryMPSCache, v; kwargs...)
  pv = planargraph_partitionvertex(bmpsc, v)
  g = subgraph(unpartitioned_graph(ppg(bmpsc)), planargraph_vertices(bmpsc, pv))
  return update(bmpsc, PartitionEdge.(post_order_dfs_edges(g, v)); kwargs...)
end

#Move the orthogonality centre one step from message tensor on pe1 to that on pe2 
function orthogonal_gauge_step(bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge; kwargs...)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  m1, m2 = only(message(bmpsc, pe1)), only(message(bmpsc, pe2))
  @assert !isempty(commoninds(m1,m2))
  left_inds = uniqueinds(m1, m2)
  m1, Y = factorize(m1, left_inds; ortho="left", kwargs...)
  m2 = m2 * Y
  set!(ms, pe1, ITensor[m1])
  set!(ms, pe2, ITensor[m2])
  return bmpsc
end

#Move the orthogonality centre via a sequence of steps between message tensors
function orthogonal_gauge_walk(bmpsc::BoundaryMPSCache, seq::Vector; kwargs...)
  for pe_pair in seq
    pe1, pe2 = PartitionEdge(parent(src(pe_pair))), PartitionEdge(parent(dst(pe_pair)))
    bmpsc = orthogonal_gauge_step(bmpsc, pe1, pe2)
  end
  return bmpsc
end

#Move the orthogonality centre to pe
function ITensorNetworks.orthogonalize(bmpsc::BoundaryMPSCache, pe::PartitionEdge; kwargs...)
  return orthogonal_gauge_walk(bmpsc, update_sequence(bmpsc, pe); kwargs...)
end

#Move the orthogonality centre from pe1 to pe2
function ITensorNetworks.orthogonalize(bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge; kwargs...)
  return orthogonal_gauge_walk(bmpsc, update_sequence(bmpsc, pe1, pe2); kwargs...)
end

function orthogonal_mps_update(bmpsc::BoundaryMPSCache, pe::PartitionEdge; niters::Int64 = 25, normalize = true)
  bmpsc = switch_messages(bmpsc, pe)
  pes = planargraph_partitionedges(bmpsc, pe)
  update_seq = vcat(pes, reverse(pes)[2:length(pes)])
  prev_v, prev_pe = nothing, nothing
  message_update = ms -> default_message_update(ms; normalize)
  for i in 1:niters
    for update_pe in update_seq
      cur_v = parent(src(update_pe))
      bmpsc = !isnothing(prev_pe) ? orthogonalize(bmpsc, reverse(prev_pe), reverse(update_pe)) : orthogonalize(bmpsc, reverse(update_pe))
      bmpsc = !isnothing(prev_v) ? partition_update(bmpsc, prev_v, cur_v; message_update) : partition_update(bmpsc, cur_v; message_update)
      me = update_message(bmpsc, update_pe; message_update)
      bmpsc = set_message(bmpsc, reverse(update_pe), dag.(me))
      prev_v, prev_pe = cur_v, update_pe
    end
  end
  return switch_messages(bmpsc, pe)
end 

function orthogonal_mps_update(bmpsc::BoundaryMPSCache, pes::Vector{<:PartitionEdge} = default_edge_sequence(ppg(bmpsc)); maxiter::Int64 = 1, kwargs...)
  bmpsc = copy(bmpsc)
  for i in 1:maxiter
    for pe in pes
      bmpsc = orthogonal_mps_update(bmpsc, pe; kwargs...)
    end
  end
  return bmpsc
end

function ITensorNetworks.environment(bmpsc::BoundaryMPSCache, verts::Vector; kwargs...)
  vs = parent.((partitionvertices(bp_cache(bmpsc), verts)))
  pv = only(planargraph_partitionvertices(bmpsc, vs))
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
    :default_edge_sequence
  ]
    @eval begin
      function $f(bmpsc::BoundaryMPSCache, args...; kwargs...)
        return $f(bp_cache(bmpsc), args...; kwargs...)
      end
    end
end

#Wrap around beliefpropagationcache
for f in [
  :update
]
  @eval begin
    function $f(bmpsc::BoundaryMPSCache, args...; kwargs...)
      bpc = $f(bp_cache(bmpsc), args...; kwargs...)
      return BoundaryMPSCache(bpc, ppg(bmpsc))
    end
  end
end

function NamedGraphs.GraphsExtensions.add_edges(pg::PartitionedGraph, pes::Vector{<:PartitionEdge})
  g = partitioned_graph(pg)
  g = add_edges(g, parent.(pes))
  return PartitionedGraph(unpartitioned_graph(pg), g, partitioned_vertices(pg), which_partition(pg))
end

function NamedGraphs.GraphsExtensions.add_edges(bpc::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
  pg = add_edges(partitioned_tensornetwork(bpc), pes)
  return BeliefPropagationCache(pg, messages(bpc), default_message(bpc))
end

function insert_missing_planar_edges(bpc::BeliefPropagationCache; sort_f = v -> first(v))
  pg = partitioned_graph(bpc)
  partitions = unique(sort_f.(collect(vertices(pg))))
  es_to_add = PartitionEdge[]
  for p in partitions
      vs = sort(filter(v -> sort_f(v) == p, collect(vertices(pg))))
      for i in 1:(length(vs) - 1)
          if vs[i] ∉ neighbors(pg, vs[i+1])
              push!(es_to_add, PartitionEdge(NamedEdge(vs[i] => vs[i+1])))
          end
      end
  end
  return add_edges(bpc, es_to_add)
end