using NamedGraphs: NamedGraphs
using NamedGraphs.GraphsExtensions: add_edges
using ITensorNetworks: ITensorNetworks, BeliefPropagationCache, region_scalar
using ITensorNetworks.ITensorsExtensions: map_diag
using ITensorMPS: ITensorMPS, orthogonalize
using NamedGraphs.PartitionedGraphs: partitioned_graph, PartitionVertex, partitionvertex, partitioned_vertices,
  which_partition
using SplitApplyCombine: group
using ITensors: commoninds
using LinearAlgebra: pinv

pair(pe::PartitionEdge) = parent(src(pe)) => parent(dst(pe))

#Return a sequence of pairs to go from item1 to item2 in an ordered_list
function pair_sequence(ordered_list::Vector, item1, item2)
  item1_pos, item2_pos = only(findall(x->x==item1, ordered_list)), only(findall(x->x==item2, ordered_list))
  item1_pos < item2_pos && return [ordered_list[i] => ordered_list[i+1] for i in item1_pos:(item2_pos-1)]
  return [ordered_list[i] => ordered_list[i-1] for i in item1_pos:-1:(item2_pos+1)]
end

struct BoundaryMPSCache{BPC,PG}
    bp_cache::BPC
    partitionedplanargraph::PG
end

bp_cache(bmpsc::BoundaryMPSCache) = bmpsc.bp_cache
partitionedplanargraph(bmpsc::BoundaryMPSCache) = bmpsc.partitionedplanargraph
ppg(bmpsc) = partitionedplanargraph(bmpsc)
planargraph(bmpsc::BoundaryMPSCache) = unpartitioned_graph(partitionedplanargraph(bmpsc))
tensornetwork(bmpsc::BoundaryMPSCache) = tensornetwork(bp_cache(bmpsc))
partitioned_tensornetwork(bmpsc::BoundaryMPSCache) = partitioned_tensornetwork(bp_cache(bmpsc))

default_edge_sequence(bmpsc::BoundaryMPSCache) = pair.(default_edge_sequence(ppg(bmpsc)))
default_bp_maxiter(bmpsc::BoundaryMPSCache) = default_bp_maxiter(partitioned_graph(ppg(bmpsc)))
default_mps_fit_kwargs(bmpsc::BoundaryMPSCache) = (; niters = 25, tolerance = 1e-10)

function Base.copy(bmpsc::BoundaryMPSCache)
  return BoundaryMPSCache(copy(bp_cache(bmpsc)), copy(ppg(bmpsc)))
end

planargraph_vertices(bmpsc::BoundaryMPSCache, partition::Int64) = vertices(ppg(bmpsc), PartitionVertex(partition))
planargraph_partition(bmpsc::BoundaryMPSCache, vertex) = parent(partitionvertex(ppg(bmpsc), vertex))
planargraph_partitions(bmpsc::BoundaryMPSCache, verts) = parent.(partitionvertices(ppg(bmpsc), verts))
planargraph_partitionpair(bmpsc::BoundaryMPSCache, pe::PartitionEdge) = pair(partitionedge(ppg(bmpsc), parent(pe)))

function BoundaryMPSCache(bpc::BeliefPropagationCache; sort_f::Function = v -> first(v),
  message_rank::Int64=1)
  bpc = insert_pseudo_planar_edges(bpc; sort_f)
  planar_graph = partitioned_graph(bpc)
  vertex_groups = group(sort_f, collect(vertices(planar_graph)))
  ppg = PartitionedGraph(planar_graph, vertex_groups)
  bmpsc = BoundaryMPSCache(bpc, ppg)
  return set_interpartition_messages(bmpsc, message_rank)
end

#Get all partitionedges within a column/row, ordered top to bottom
function planargraph_partitionedges(bmpsc::BoundaryMPSCache, partition::Int64)
  vs = sort(planargraph_vertices(bmpsc, partition))
  return PartitionEdge.([vs[i] => vs[i+1] for i in 1:(length(vs)-1)])
end

#Get the sequence of pairs partitionedges that need to be updated to move the MPS gauge from pe1 to pe2
function mps_gauge_update_sequence(bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge)
  ppgpe1, ppgpe2 = planargraph_partitionpair(bmpsc, pe1), planargraph_partitionpair(bmpsc, pe2)
  @assert ppgpe1 == ppgpe2
  pes = planargraph_partitionpair_partitionedges(bmpsc, ppgpe1)
  return pair_sequence(pes, pe1, pe2)
end

#Get the sequence of pairs partitionedges that need to be updated to move the MPS gauge onto pe1
function mps_gauge_update_sequence(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  ppgpe = planargraph_partitionpair(bmpsc, pe)
  pes =  planargraph_partitionpair_partitionedges(bmpsc, ppgpe)
  return vcat(mps_gauge_update_sequence(bmpsc, last(pes), pe), mps_gauge_update_sequence(bmpsc, first(pes), pe))
end

#Get all partitionedges between the pair of neighboring partitions, sorted top to bottom
#TODO: Bring in line with NamedGraphs change
function planargraph_partitionpair_partitionedges(bmpsc::BoundaryMPSCache, pe::Pair)
  pg = planargraph(bmpsc)
  src_vs, dst_vs = planargraph_vertices(bmpsc, first(pe)), planargraph_vertices(bmpsc, last(pe))
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

#Initialise all the message tensors for the pair of neighboring partitions, with virtual rank given by message rank
function set_interpartition_messages(bmpsc::BoundaryMPSCache, pe::Pair, message_rank::Int64)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  pes = planargraph_partitionpair_partitionedges(bmpsc, pe)
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

#Initialise all the inerpartition message tensors with virtual rank given by message rank
function set_interpartition_messages(bmpsc::BoundaryMPSCache, message_rank::Int64 = 1)
  bmpsc = copy(bmpsc)
  pes = partitionedges(ppg(bmpsc))
  for pe in vcat(pes, reverse(reverse.(pes)))
    bmpsc = set_interpartition_messages(bmpsc, parent(src(pe)) => parent(dst(pe)), message_rank)
  end
  return bmpsc
end

#Switch the message tensors from column/row  i -> i + 1 with those from i + 1 -> i
function switch_messages(bmpsc::BoundaryMPSCache, pe::Pair)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  pes = planargraph_partitionpair_partitionedges(bmpsc, pe)
  for pe_i in pes
    me, mer = message(bmpsc, pe_i), message(bmpsc, reverse(pe_i))
    set!(ms, pe_i, dag.(mer))
    set!(ms, reverse(pe_i), dag.(me))
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

#Update all messages within a partition towards v
function partition_update(bmpsc::BoundaryMPSCache, v; kwargs...)
  pv = planargraph_partition(bmpsc, v)
  g = subgraph(unpartitioned_graph(ppg(bmpsc)), planargraph_vertices(bmpsc, pv))
  return update(bmpsc, PartitionEdge.(post_order_dfs_edges(g, v)); kwargs...)
end

#Move the orthogonality centre one step on an interpartition from the message tensor on pe1 to that on pe2 
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

#Move the orthogonality centre on an interpartition via a sequence of steps between message tensors
function orthogonal_gauge_walk(bmpsc::BoundaryMPSCache, seq::Vector; kwargs...)
  for (pe1, pe2) in seq
    bmpsc = orthogonal_gauge_step(bmpsc, pe1, pe2)
  end
  return bmpsc
end

#Move the biorthogonality centre on an interpartition via a sequence of steps between message tensors
function biorthogonal_gauge_walk(bmpsc::BoundaryMPSCache, seq::Vector; kwargs...)
  for (pe1, pe2) in seq
    bmpsc = biorthogonal_gauge_step(bmpsc, pe1, pe2)
  end
  return bmpsc
end

#Move the orthogonality centre on an interpartition to the message tensor on pe
function ITensorMPS.orthogonalize(bmpsc::BoundaryMPSCache, pe::PartitionEdge; kwargs...)
  return orthogonal_gauge_walk(bmpsc, mps_gauge_update_sequence(bmpsc, pe); kwargs...)
end

#Move the orthogonality centre on an interpartition from the message tensor on pe1 to that on pe2 
function ITensorMPS.orthogonalize(bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge; kwargs...)
  return orthogonal_gauge_walk(bmpsc, mps_gauge_update_sequence(bmpsc, pe1, pe2); kwargs...)
end

#Move the biorthogonality centre on an interpartition to the message tensor on pe
function biorthogonalize(bmpsc::BoundaryMPSCache, pe::PartitionEdge; kwargs...)
  return biorthogonal_gauge_walk(bmpsc, mps_gauge_update_sequence(bmpsc, pe); kwargs...)
end

#Move the biorthogonality centre on an interpartition from the message tensor on pe1 to that on pe2 
function biorthogonalize(bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge; kwargs...)
  return biorthogonal_gauge_walk(bmpsc, mps_gauge_update_sequence(bmpsc, pe1, pe2); kwargs...)
end

#Update all the message tensors on an interpartition via an orthogonal fitting procedure
function orthogonal_mps_update(bmpsc::BoundaryMPSCache, pe::Pair; niters::Int64 = 25, tolerance = 1e-10, normalize = true)
  bmpsc = switch_messages(bmpsc, pe)
  pes = planargraph_partitionpair_partitionedges(bmpsc, pe)
  update_seq = vcat(pes, reverse(pes)[2:length(pes)])
  prev_v, prev_pe = nothing, nothing
  prev_costfunction = 0
  for i in 1:niters
    costfunction = 0
    for update_pe in update_seq
      cur_v = parent(src(update_pe))
      bmpsc = !isnothing(prev_pe) ? orthogonalize(bmpsc, reverse(prev_pe), reverse(update_pe)) : orthogonalize(bmpsc, reverse(update_pe))
      bmpsc = !isnothing(prev_v) ? partition_update(bmpsc, prev_v, cur_v; message_update = ms -> default_message_update(ms; normalize = false)) : partition_update(bmpsc, cur_v; message_update = ms -> default_message_update(ms; normalize = false))
      me = update_message(bmpsc, update_pe; message_update = ms -> default_message_update(ms; normalize))
      costfunction += region_scalar(bp_cache(bmpsc), src(update_pe)) / norm(me)
      bmpsc = set_message(bmpsc, reverse(update_pe), dag.(me))
      prev_v, prev_pe = cur_v, update_pe
    end
    epsilon = abs(costfunction - prev_costfunction) / length(update_seq)
    if !isnothing(tolerance) && epsilon < tolerance
      return switch_messages(bmpsc, pe)
    else
      prev_costfunction = costfunction
    end
  end
  return switch_messages(bmpsc, pe)
end 

#Update all the message tensors on an interpartition via a biorthogonal fitting procedure
function biorthogonal_mps_update(bmpsc::BoundaryMPSCache, pe::Pair; normalize = true)
  pes = planargraph_partitionpair_partitionedges(bmpsc, pe)
  update_seq = pes
  prev_v, prev_pe = nothing, nothing
  for update_pe in update_seq
    cur_v = parent(src(update_pe))
    bmpsc = !isnothing(prev_pe) ? biorthogonalize(bmpsc, prev_pe, update_pe) : biorthogonalize(bmpsc, update_pe)
    bmpsc = !isnothing(prev_v) ? partition_update(bmpsc, prev_v, cur_v; message_update = ms -> default_message_update(ms; normalize = false)) : partition_update(bmpsc, cur_v; message_update = ms -> default_message_update(ms; normalize = false))
    me = only(update_message(bmpsc, update_pe; message_update = ms -> default_message_update(ms; normalize)))
    mer = only(message(bmpsc, reverse(update_pe)))
    me = replaceinds(me, noncommoninds(me, mer), noncommoninds(mer, me))
    bmpsc = set_message(bmpsc, update_pe, ITensor[me])
    prev_v, prev_pe = cur_v, update_pe
  end

  return bmpsc
end 


"""
More generic interface for update, with default params
"""
function update(
  bmpsc::BoundaryMPSCache;
  pes=default_edge_sequence(bmpsc),
  maxiter=default_bp_maxiter(bmpsc),
  mps_fit_kwargs = default_mps_fit_kwargs(bmpsc)
)
  if isnothing(maxiter)
    error("You need to specify a number of iterations for Boundary MPS!")
  end
  for i in 1:maxiter
    for pe in pes
      bmpsc = orthogonal_mps_update(bmpsc, pe; mps_fit_kwargs...)
    end
  end
  return bmpsc
end

"""
More generic interface for update, with default params
"""
function biorthognal_update(
  bmpsc::BoundaryMPSCache;
  pes=default_edge_sequence(bmpsc),
  maxiter=25,
)
  if isnothing(maxiter)
    error("You need to specify a number of iterations for Boundary MPS!")
  end
  for i in 1:maxiter
    for pe in pes
      bmpsc = biorthogonal_mps_update(bmpsc, pe)
    end
  end
  return bmpsc
end




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
for f in [
  :update_factors,
  :update_factor,
]
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

#Add partition edges that may not have meaning in the underlying graph
function add_pseudo_partitionedges(pg::PartitionedGraph, pes::Vector{<:PartitionEdge})
  g = partitioned_graph(pg)
  g = add_edges(g, parent.(pes))
  return PartitionedGraph(unpartitioned_graph(pg), g, partitioned_vertices(pg), which_partition(pg))
end

#Add partition edges that may not have meaning in the underlying graph
function add_pseudo_partitionedges(bpc::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
  pg = add_pseudo_partitionedges(partitioned_tensornetwork(bpc), pes)
  return BeliefPropagationCache(pg, messages(bpc), default_message(bpc))
end

#Add partition edges necessary to connect up all vertices in a partition in the planar graph created by the sort function
function insert_pseudo_planar_edges(bpc::BeliefPropagationCache; sort_f = v -> first(v))
  pg = partitioned_graph(bpc)
  partitions = unique(sort_f.(collect(vertices(pg))))
  pseudo_edges = PartitionEdge[]
  for p in partitions
      vs = sort(filter(v -> sort_f(v) == p, collect(vertices(pg))))
      for i in 1:(length(vs) - 1)
          if vs[i] ∉ neighbors(pg, vs[i+1])
              push!(pseudo_edges, PartitionEdge(NamedEdge(vs[i] => vs[i+1])))
          end
      end
  end
  return add_pseudo_partitionedges(bpc, pseudo_edges)
end


function biorthogonal_gauge_step(bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge; regularization = 0)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)

  m1, m1r = only(message(bmpsc, pe1)), only(message(bmpsc, reverse(pe1)))
  m2, m2r = only(message(bmpsc, pe2)), only(message(bmpsc, reverse(pe2)))
  top_cind, bottom_cind = commonind(m1, m2), commonind(m1r, m2r)
  m1_siteinds, m2_siteinds = commoninds(m1, m1r), commoninds(m2, m2r)
  top_ncind, bottom_ncind = setdiff(inds(m1), [m1_siteinds; top_cind]), setdiff(inds(m1r), [m1_siteinds; bottom_cind])

  E = isempty(top_ncind) ? m1 * m1r : m1 * replaceind(m1r, only(bottom_ncind), only(top_ncind))
  U, S, V = svd(E, bottom_cind; alg = "recursive")
  S_sqrtinv = map_diag(x -> pinv(sqrt(x + regularization)), S)
  S_sqrt = map_diag(x -> sqrt(x + regularization), S)

  set!(ms, pe1, ITensor[replaceind((m1 * dag(V)) * S_sqrtinv, commonind(U, S), top_cind)])
  set!(ms, reverse(pe1), ITensor[replaceind((m1r * dag(U)) * S_sqrtinv, commonind(V, S), bottom_cind)])
  set!(ms, pe2, ITensor[replaceind((m2 * V) * S_sqrt, commonind(U, S), top_cind)])
  set!(ms, reverse(pe2), ITensor[replaceind((m2r * U) * S_sqrt, commonind(V, S), bottom_cind)])

  @show inds(only(ms[pe1])), inds(only(ms[reverse(pe1)]))
  @show inds(only(ms[pe2])), inds(only(ms[reverse(pe2)]))
  return bmpsc
end

function biorthogonalize_walk(ψ::ITensorNetwork, ϕ::ITensorNetwork, path::Vector{<:NamedEdge};
  regularization = 0)
  ψ, ϕ = copy(ψ), copy(ϕ)
  ep = first(path)
  for e in path
      vsrc, vdst = src(e), dst(e)
      top_cind = commonind(ψ[vsrc], ψ[vdst])
      bottom_cind = commonind(ϕ[vsrc], ϕ[vdst])
      vsrc_siteinds = commoninds(ψ[vsrc], ϕ[vsrc])
      top_src_ncind = setdiff(inds(ψ[vsrc]), [vsrc_siteinds; top_cind])
      bottom_src_ncind = setdiff(inds(ϕ[vsrc]), [vsrc_siteinds; bottom_cind])
      if isempty(top_src_ncind)
          E = ψ[vsrc] * ϕ[vsrc]
      else
          E = ψ[vsrc] * replaceind(ϕ[vsrc], only(bottom_src_ncind), only(top_src_ncind))
      end

      U, S, V = svd(E, bottom_cind; alg = "recursive")
      S_sqrtinv = map_diag(x -> pinv(sqrt(x + regularization)), S)
      S_sqrt = map_diag(x -> sqrt(x + regularization), S)
      ψ[vsrc] = replaceind((ψ[vsrc] * dag(V)) * S_sqrtinv, commonind(U, S), top_cind)
      ϕ[vsrc] = replaceind((ϕ[vsrc] * dag(U)) * S_sqrtinv, commonind(V, S), bottom_cind)

      ψ[vdst] = replaceind((ψ[vdst] * V) * S_sqrt, commonind(U, S), top_cind)
      ϕ[vdst] = replaceind((ϕ[vdst] * U) * S_sqrt, commonind(V, S), bottom_cind)

  end

  return ψ, ϕ
end