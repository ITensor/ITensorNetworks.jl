using ITensorNetworks: ITensorNetworks, BeliefPropagationCache
using NamedGraphs.PartitionedGraphs: PartitionedGraph

function add_partitionedges(pg::PartitionedGraph, pes::Vector{<:PartitionEdge})
  g = partitioned_graph(pg)
  g = add_edges(g, parent.(pes))
  return PartitionedGraph(
    unpartitioned_graph(pg), g, partitioned_vertices(pg), which_partition(pg)
  )
end

function add_partitionedges(bpc::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
  pg = add_partitionedges(partitioned_tensornetwork(bpc), pes)
  return BeliefPropagationCache(pg, messages(bpc))
end

#Add partition edges necessary to connect up all vertices in a partition in the planar graph created by the sort function
function insert_pseudo_planar_edges(
  bpc::BeliefPropagationCache; grouping_function=v -> first(v)
)
  pg = partitioned_graph(bpc)
  partitions = unique(grouping_function.(collect(vertices(pg))))
  pseudo_edges = PartitionEdge[]
  for p in partitions
    vs = sort(filter(v -> grouping_function(v) == p, collect(vertices(pg))))
    for i in 1:(length(vs) - 1)
      if vs[i] ∉ neighbors(pg, vs[i + 1])
        push!(pseudo_edges, PartitionEdge(NamedEdge(vs[i] => vs[i + 1])))
      end
    end
  end
  return add_partitionedges(bpc, pseudo_edges)
end

pair(pe::PartitionEdge) = parent(src(pe)) => parent(dst(pe))
