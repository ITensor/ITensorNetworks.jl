using ITensorNetworks: ITensorNetworks, BeliefPropagationCache
using NamedGraphs.PartitionedGraphs: PartitionedGraph

#Add partition edges that may not have meaning in the underlying graph
function add_partitionedges(pg::PartitionedGraph, pes::Vector{<:PartitionEdge})
  g = partitioned_graph(pg)
  g = add_edges(g, parent.(pes))
  return PartitionedGraph(
    unpartitioned_graph(pg), g, partitioned_vertices(pg), which_partition(pg)
  )
end

#Add partition edges that may not have meaning in the underlying graph
function add_partitionedges(bpc::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
  pg = add_partitionedges(partitioned_tensornetwork(bpc), pes)
  return BeliefPropagationCache(pg, messages(bpc), default_message(bpc))
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

#Return a sequence of pairs to go from item1 to item2 in an ordered_list
function pair_sequence(ordered_list::Vector, item1, item2)
  item1_pos, item2_pos = only(findall(x -> x == item1, ordered_list)),
  only(findall(x -> x == item2, ordered_list))
  item1_pos < item2_pos &&
    return [ordered_list[i] => ordered_list[i + 1] for i in item1_pos:(item2_pos - 1)]
  return [ordered_list[i] => ordered_list[i - 1] for i in item1_pos:-1:(item2_pos + 1)]
end
