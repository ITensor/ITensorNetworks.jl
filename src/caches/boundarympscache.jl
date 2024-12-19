using NamedGraphs: NamedGraphs
using ITensorNetworks: ITensorNetworks
using NamedGraphs.PartitionedGraphs: partitioned_graph

struct BoundaryMPSCache{BPC,G, D}
    bp_cache::BPC
    gauges::G
    partitions::D
end

bp_cache(bmpsc::BoundaryMPSCache) = bmpsc.bp_cache
gauges(bmpsc::BoundaryMPSCache) = bmpsc.gauges
partitions(bmpsc::BoundaryMPSCache) = bmpsc.partitions
planargraph(bmpsc::BoundaryMPSCache) = partitioned_graph(bp_cache(bmpsc))

ITensorNetworks.partitionvertices(bmpsc::BoundaryMPSCache, partition::Int64) = partitions(bmpsc)[partition]

function get_partition(bmpsc::BoundaryMPSCache, v)
  return only(filter(pv -> v ∈ partitionvertices(bmpsc, pv), keys(partitions(bmpsc))))
end

function BoundaryMPSCache(bpc::BeliefPropagationCache; sort_f = v -> first(v))
    planar_graph = partitioned_graph(bpc)
    #TODO: Make sure these are sorted
    vertex_groups = group(sort_f, collect(vertices(planar_graph)))

    gauges = Dictionary(keys(vertex_groups), ["Nothing" for pv in keys(vertex_groups)])
    return BoundaryMPSCache(bpc, gauges, vertex_groups)
end

#Get all partitionedges within a partition, sorted top to bottom 
function ITensorNetworks.partitionedges(bmpsc::BoundaryMPSCache, partition::Int64)
  vs = partitionvertices(bmpsc, partition)
  return PartitionEdge.([vs[i] => vs[i+1] for i in 1:(length(vs)-1)])
end

#Update all messages flowing within a partition
function partition_update(bmpsc::BoundaryMPSCache, pv::PartitionVertex; kwargs...)
  edges = partitionedges(bmpsc, pv)
  return update(bmpsc, vcat(edges, reverse(reverse.(edges))); kwargs...)
end

#Edges between v1 and v2 within a partition
function update_sequence(bmpsc::BoundaryMPSCache, v1, v2)
  pv1, pv2 = get_partition(bmpsc, v1), get_partition(bmpsc, v2)
  @assert pv1 == pv2
  vs = partitionvertices(bmpsc, pv1)
  v1_pos, v2_pos = only(findall(x->x==v1, vs)), only(findall(x->x==v2, vs))
  v1_pos < v2_pos && return PartitionEdge.([vs[i] => vs[i+1] for i in v1_pos:(v2_pos-1)])
  return PartitionEdge.([vs[i] => vs[i-1] for i in v2_pos:(v1_pos+1)])
end

#Update all messages flowing within a partition from v1 to v2
function partition_update(bmpsc::BoundaryMPSCache, v1, v2; kwargs...)
  return update(bmpsc, update_sequence(bmpsc, v1, v2); kwargs...)
end

#Needed for implementation, forward from beliefpropagationcache
for f in [
    :messages,
    :message,
    :update_message,
    :update,
  ]
    @eval begin
      function $f(bmpsc::BoundaryMPSCache, args...; kwargs...)
        return $f(bp_cache(bmpsc), args...; kwargs...)
      end
    end
end