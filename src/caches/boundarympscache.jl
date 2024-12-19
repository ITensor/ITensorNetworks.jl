using NamedGraphs: NamedGraphs
using ITensorNetworks: ITensorNetworks, BeliefPropagationCache
using NamedGraphs.PartitionedGraphs: partitioned_graph, PartitionVertex, partitionvertex
using SplitApplyCombine: group

struct BoundaryMPSCache{BPC,G, PG}
    bp_cache::BPC
    gauges::G
    partitionedplanargraph::PG
end

bp_cache(bmpsc::BoundaryMPSCache) = bmpsc.bp_cache
gauges(bmpsc::BoundaryMPSCache) = bmpsc.gauges
partitionedplanargraph(bmpsc::BoundaryMPSCache) = bmpsc.partitionedplanargraph
ppg(bmpsc) = partitionedplanargraph(bmpsc)
planargraph(bmpsc::BoundaryMPSCache) = unpartitioned_graph(partitionedplanargraph(bmpsc))

function Base.copy(bmpsc::BoundaryMPSCache)
  return BoundaryMPSCache(copy(bp_cache(bmpsc)), copy(gauges(bmpsc)), copy(ppg(bmpsc)))
end

planargraph_vertices(bmpsc::BoundaryMPSCache, pv::PartitionVertex) = vertices(ppg(bmpsc), pv)
planargraph_partitionvertex(bmpsc::BoundaryMPSCache, vertex) = partitionvertex(ppg(bmpsc), vertex)

function BoundaryMPSCache(bpc::BeliefPropagationCache; sort_f = v -> first(v))
    planar_graph = partitioned_graph(bpc)
    vertex_groups = group(sort_f, collect(vertices(planar_graph)))
    ppg = PartitionedGraph(planar_graph, vertex_groups)
    gauges = Dictionary(partitionvertices(ppg), ["Nothing" for pv in partitionvertices(ppg)])
    return BoundaryMPSCache(bpc, gauges, ppg)
end

#Get all partitionedges within a partition, sorted top to bottom 
function planargraph_partitionedges(bmpsc::BoundaryMPSCache, pv::PartitionVertex)
  vs = sort(planargraph_vertices(bmpsc, pv))
  return PartitionEdge.([vs[i] => vs[i+1] for i in 1:(length(vs)-1)])
end

#Edges between v1 and v2 within a partition
function update_sequence(bmpsc::BoundaryMPSCache, v1, v2)
  pv1, pv2 = planargraph_partitionvertex(bmpsc, v1), planargraph_partitionvertex(bmpsc, v2)
  @assert pv1 == pv2
  vs = planargraph_vertices(bmpsc, pv1)
  v1_pos, v2_pos = only(findall(x->x==v1, vs)), only(findall(x->x==v2, vs))
  v1_pos < v2_pos && return PartitionEdge.([vs[i] => vs[i+1] for i in v1_pos:(v2_pos-1)])
  return PartitionEdge.([vs[i] => vs[i-1] for i in v1_pos:(v2_pos+1)])
end

#Get all partitionedges from p1 to p2, flowing from top to bottom 
#TODO: Bring in line with NamedGraphs change
function planargraph_partitionedges(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pg = planargraph(bmpsc)
  src_vs, dst_vs = planargraph_vertices(bmpsc, src(pe)), planargraph_vertices(bmpsc, dst(pe))
  es = filter(x -> !isempty(last(x)), [src_v => intersect(neighbors(pg, src_v), dst_vs) for src_v in src_vs])
  es = map(x -> first(x) => only(last(x)), es)
  return PartitionEdge.(NamedEdge.(es))
end

function set_message(bmpsc::BoundaryMPSCache, pe::PartitionEdge, m::Vector{ITensor})
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  set!(ms, pe, m)
  return bmpsc
end

function set_messages(bmpsc::BoundaryMPSCache, pe::PartitionEdge; message_rank::Int64 = 1)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  pes = planargraph_partitionedges(bmpsc, pe)
  prev_virtual_ind = nothing
  for (i, pg_pe) in enumerate(pes)
    siteinds = linkinds(bmpsc, pg_pe)
    next_virtual_index = i != length(pes) ? Index(message_rank, "m$(i)$(i+1)") : nothing
    inds = filter(x -> !isnothing(x), [siteinds; prev_virtual_ind; next_virtual_index])
    set!(ms, pg_pe, ITensor[ITensor(1.0, inds)])
    prev_virtual_ind = next_virtual_index
  end
  return bmpsc
end

function set_messages(bmpsc::BoundaryMPSCache; message_rank::Int64 = 1)
  bmpsc = copy(bmpsc)
  pes = partitionedges(ppg(bmpsc))
  for pe in vcat(pes, reverse(reverse.(pes)))
    bmpsc = set_messages(bmpsc, pe; message_rank)
  end
  return bmpsc
end

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
  edges = planargraph_partitionedges(bmpsc, pv)
  return update(bmpsc, vcat(edges, reverse(reverse.(edges))); kwargs...)
end

#Update all messages flowing within a partition from v1 to v2
function partition_update(bmpsc::BoundaryMPSCache, v1, v2; kwargs...)
  v1 == v2 && return bmpsc
  isnothing(v1) && return partition_update(bmpsc, planargraph_partitionvertex(bmpsc, v2))
  return update(bmpsc, update_sequence(bmpsc, v1, v2); kwargs...)
end

function mps_update(bmpsc::BoundaryMPSCache, pe::PartitionEdge; niters::Int64 = 1)
  bmpsc = switch_messages(bmpsc, pe)
  pes = planargraph_partitionedges(bmpsc, pe)
  update_seq = vcat(pes, reverse(pes))
  prev_v = nothing
  for i in 1:niters
    for update_pe in update_seq
      cur_v = parent(src(update_pe))
      #TODO: Move gauge centre to update_pe

      bmpsc = partition_update(bmpsc, prev_v, cur_v)
      me = update_message(bmpsc, update_pe)
      bmpsc = set_message(bmpsc, reverse(update_pe), dag.(me))
      prev_v = cur_v
    end
  end
  return switch_messages(bmpsc, pe)
end 

function mps_update(bmpsc::BoundaryMPSCache, pes::Vector{<:PartitionEdge} = default_edge_sequence(ppg(bmpsc)); maxiter::Int64 = 1, niters::Int64 = 1)
  bmpsc = copy(bmpsc)
  for i in 1:maxiter
    for pe in pes
      bmpsc = mps_update(bmpsc, pe; niters)
    end
  end
  return bmpsc
end

#Forward onto beliefpropagationcache
for f in [
    :messages,
    :message,
    :update_message,
    :(ITensorNetworks.linkinds),
    :default_edge_sequence,
    :environment
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
      return BoundaryMPSCache(bpc, gauges(bmpsc), ppg(bmpsc))
    end
  end
end