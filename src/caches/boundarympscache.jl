using NamedGraphs: NamedGraphs
using ITensorNetworks: ITensorNetworks
using NamedGraphs.PartitionedGraphs: partitioned_graph

struct BoundaryMPSCache{BPC,G, D}
    bp_cache::BPC
    gauges::G
    partitionedvertices::D
end

bp_cache(bmpsc::BoundaryMPSCache) = bmpsc.bp_cache
gauges(bmpsc::BoundaryMPSCache) = bmpsc.gauges
partitionedplanargraph(bmpsc::BoundaryMPSCache) = bmpsc.partitionedplanargraph

function BoundaryMPSCache(bpc::BeliefPropagationCache; sort_f = v -> first(v))
    planar_graph = partitioned_graph(bpc)
    vertex_groups = group(sort_f, collect(vertices(planar_graph)))
    gauges = Dictionary(keys(vertex_groups), ["Nothing" for pv in keys(vertex_groups)])
    return BoundaryMPSCache(bpc, gauges, vertex_groups)
end

#Get all partitionedges within a column / row partition
function ITensorNetworks.partitionedges(bmpsc::BoundaryMPSCache, pv::PartitionVertex)
  pg = partitionedplanargraph(bmpsc)
  vs = sort(vertices(pg, pv))
  return PartitionEdge.([vs[i] => vs[i+1] for i in 1:length(vs)])
end

# #Get all partitionedges between rows/ columns
function ITensorNetworks.partitionedges(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pg = partitionedplanargraph(bmpsc)
  pv_src_verts, pv_dst_verts = vertices(pg, src(pe)), vertices(pg, dst(pe))
  es = vcat(edges(pg), reverse.(edges(pg)))
  return PartitionEdge.(filter(e -> src(e) ∈ pv_src_verts && dst(e) ∈ pv_dst_verts, es))
end

# function partition_update(bmpsc::BoundaryMPSCache, pv::PartitionVertex)

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