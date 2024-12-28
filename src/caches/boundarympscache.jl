using NamedGraphs: NamedGraphs
using NamedGraphs.GraphsExtensions: add_edges
using ITensorNetworks: ITensorNetworks, BeliefPropagationCache
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
planargraph_partitionedge(bmpsc::BoundaryMPSCache, pe::PartitionEdge) = partitionedge(ppg(bmpsc), parent(pe))

function BoundaryMPSCache(bpc::BeliefPropagationCache; sort_f::Function = v -> first(v))
  bpc = insert_missing_planar_edges(bpc; sort_f)
  planar_graph = partitioned_graph(bpc)
  vertex_groups = group(sort_f, collect(vertices(planar_graph)))
  ppg = PartitionedGraph(planar_graph, vertex_groups)
  return BoundaryMPSCache(bpc, ppg)
end

#Get all partitionedges within a partition, sorted top to bottom 
function planargraph_partitionedges(bmpsc::BoundaryMPSCache, pv::PartitionVertex)
  vs = sort(planargraph_vertices(bmpsc, pv))
  return PartitionEdge.([vs[i] => vs[i+1] for i in 1:(length(vs)-1)])
end

#Edges between v1 and v2 within a partition
function update_sequence(bmpsc::BoundaryMPSCache, v1, v2)
  v1 == v2 && return PartitionEdge[]
  pv1, pv2 = planargraph_partitionvertex(bmpsc, v1), planargraph_partitionvertex(bmpsc, v2)
  @assert pv1 == pv2
  vs = sort(planargraph_vertices(bmpsc, pv1))
  v1_pos, v2_pos = only(findall(x->x==v1, vs)), only(findall(x->x==v2, vs))
  v1_pos < v2_pos && return PartitionEdge.([vs[i] => vs[i+1] for i in v1_pos:(v2_pos-1)])
  return PartitionEdge.([vs[i] => vs[i-1] for i in v1_pos:-1:(v2_pos+1)])
end

#Edges toward v within a partition
function update_sequence(bmpsc::BoundaryMPSCache, v)
  pv = planargraph_partitionvertex(bmpsc, v)
  vs = sort(planargraph_vertices(bmpsc, pv))
  seq = vcat(update_sequence(bmpsc, last(vs), v), update_sequence(bmpsc, first(vs), v))
  return seq
end

#Edges between pe1 and pe2 along an interpartition
function update_sequence(bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge)
  ppgpe1, ppgpe2 = planargraph_partitionedge(bmpsc, pe1), planargraph_partitionedge(bmpsc, pe2)
  @assert ppgpe1 == ppgpe2
  #TODO: Sort these top to bottom
  pes = planargraph_partitionedges(bmpsc, ppgpe1)
  pes = sort(pes; by = x -> src(parent(x)))
  pe1_pos, pe2_pos = only(findall(x->x==pe1, pes)), only(findall(x->x==pe2, pes))
  pe1_pos < pe2_pos && return PartitionEdge.([parent(pes[i]) => parent(pes[i+1]) for i in pe1_pos:(pe2_pos-1)])
  return PartitionEdge.([parent(pes[i]) => parent(pes[i-1]) for i in pe1_pos:-1:(pe2_pos+1)])
end

#Edges toward pe1 along an interpartition
function update_sequence(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  ppgpe = planargraph_partitionedge(bmpsc, pe)
  #TODO: Sort these top to bottom
  pes =  planargraph_partitionedges(bmpsc, ppgpe)
  pes = sort(pes; by = x -> src(parent(x)))
  return vcat(update_sequence(bmpsc, last(pes), pe), update_sequence(bmpsc, first(pes), pe))
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
  pes = sort(pes; by = x -> src(parent(x)))
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
  pes = sort(pes; by = x -> src(parent(x)))
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
  return update(bmpsc, update_sequence(bmpsc, v1, v2); kwargs...)
end

#Update all messages flowing within a partition from v1 to v2
function partition_update(bmpsc::BoundaryMPSCache, v; kwargs...)
  return update(bmpsc, update_sequence(bmpsc, v); kwargs...)
end

function ortho_gauge(a::ITensor, b::ITensor; kwargs...)
  @assert !isempty(commoninds(a,b))
  left_inds = uniqueinds(a, b)
  X, Y = factorize(a, left_inds; ortho="left", kwargs...)
  return X, b*Y
end

function gauge_move(bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge; kwargs...)
  if length(message(bmpsc, pe1)) == 1
    m1 = only(message(bmpsc, pe1))
  else
    m1 = contract(message(bmpsc, pe1); sequence = "automatic")
  end
  if length(message(bmpsc, pe2)) == 1
    m2 = only(message(bmpsc, pe2))
  else
    m2 = contract(message(bmpsc, pe2); sequence = "automatic")
  end
  m1, m2 = ortho_gauge(m1, m2; kwargs...)
  bmpsc = set_message(bmpsc, pe1, ITensor[m1])
  bmpsc = set_message(bmpsc, pe2, ITensor[m2])
  return bmpsc
end

function ortho_gauge(bmpsc::BoundaryMPSCache, pe::PartitionEdge; kwargs...)
  pe_seq = update_sequence(bmpsc, pe)
  for pe_pair in pe_seq
    pe1, pe2 = PartitionEdge(parent(src(pe_pair))), PartitionEdge(parent(dst(pe_pair)))
    bmpsc = gauge_move(bmpsc, pe1, pe2)
  end
  return bmpsc
end

function ortho_gauge(bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge; kwargs...)
  pe_seq = update_sequence(bmpsc, pe1, pe2)
  for pe_pair in pe_seq
    pe1, pe2 = PartitionEdge(parent(src(pe_pair))), PartitionEdge(parent(dst(pe_pair)))
    bmpsc = gauge_move(bmpsc, pe1, pe2)
  end
  return bmpsc
end


function mps_update(bmpsc::BoundaryMPSCache, pe::PartitionEdge; niters::Int64 = 1)
  bmpsc = switch_messages(bmpsc, pe)
  pes = planargraph_partitionedges(bmpsc, pe)
  pe
  update_seq = vcat(pes, reverse(pes))
  prev_v = nothing
  prev_pe = nothing
  for i in 1:niters
    for update_pe in update_seq
      cur_v = parent(src(update_pe))
      if !isnothing(prev_pe)
        bmpsc = ortho_gauge(bmpsc, reverse(prev_pe), reverse(update_pe))
      else
        bmpsc = ortho_gauge(bmpsc, reverse(update_pe))
      end
      if !isnothing(prev_v)
        bmpsc = partition_update(bmpsc, prev_v, cur_v)
      else
        bmpsc = partition_update(bmpsc, cur_v)
      end
      #TODO: This will be missing incoming messages in depleted square graphs?!?!
      me = update_message(bmpsc, update_pe)
      bmpsc = set_message(bmpsc, reverse(update_pe), dag.(me))
      prev_v = cur_v
      prev_pe = update_pe
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