using DataGraphs: AbstractDataGraph, DataGraph, edge_data, edge_data_eltype, vertex_data
using Dictionaries: Dictionary
using Graphs: AbstractGraph, add_edge!, has_edge, dst, edges, edgetype, src, vertices
using ITensors: ITensor, noncommoninds
using NamedGraphs: NamedGraph, subgraph

function _partition(g::AbstractGraph, subgraph_vertices)
  partitioned_graph = DataGraph(
    NamedGraph(eachindex(subgraph_vertices));
    vertex_data_eltype=typeof(g),
    edge_data_eltype=@NamedTuple{
      edges::Vector{edgetype(g)}, edge_data::Dictionary{edgetype(g),edge_data_eltype(g)}
    }
  )
  for v in vertices(partitioned_graph)
    partitioned_graph[v] = subgraph(g, subgraph_vertices[v])
  end
  for e in edges(g)
    s1 = findfirst_on_vertices(subgraph -> src(e) ∈ vertices(subgraph), partitioned_graph)
    s2 = findfirst_on_vertices(subgraph -> dst(e) ∈ vertices(subgraph), partitioned_graph)
    if (!has_edge(partitioned_graph, s1, s2) && s1 ≠ s2)
      add_edge!(partitioned_graph, s1, s2)
      partitioned_graph[s1 => s2] = (;
        edges=Vector{edgetype(g)}(), edge_data=Dictionary{edgetype(g),edge_data_eltype(g)}()
      )
    end
    if has_edge(partitioned_graph, s1, s2)
      push!(partitioned_graph[s1 => s2].edges, e)
      if isassigned(g, e)
        set!(partitioned_graph[s1 => s2].edge_data, e, g[e])
      end
    end
  end
  return partitioned_graph
end

"""
Find all vertices `v` such that `f(graph[v]) == true`
"""
function findall_on_vertices(f::Function, graph::AbstractDataGraph)
  return findall(f, vertex_data(graph))
end

"""
Find the vertex `v` such that `f(graph[v]) == true`
"""
function findfirst_on_vertices(f::Function, graph::AbstractDataGraph)
  return findfirst(f, vertex_data(graph))
end

"""
Find all edges `e` such that `f(graph[e]) == true`
"""
function findall_on_edges(f::Function, graph::AbstractDataGraph)
  return findall(f, edge_data(graph))
end

"""
Find the edge `e` such that `f(graph[e]) == true`
"""
function findfirst_on_edges(f::Function, graph::AbstractDataGraph)
  return findfirst(f, edge_data(graph))
end

# function subgraphs(g::AbstractSimpleGraph, subgraph_vertices)
#   return subgraphs(NamedGraph(g), subgraph_vertices)
# end

# """
#     subgraphs(g::AbstractGraph, subgraph_vertices)

# Return a collection of subgraphs of `g` defined by the subgraph
# vertices `subgraph_vertices`.
# """
# function subgraphs(g::AbstractGraph, subgraph_vertices)
#   return map(vs -> subgraph(g, vs), subgraph_vertices)
# end

# """
#     subgraphs(g::AbstractGraph; npartitions::Integer, kwargs...)

# Given a graph `g`, partition `g` into `npartitions` partitions
# or into partitions with `nvertices_per_partition` vertices per partition,
# returning a list of subgraphs.
# Try to keep all subgraphs the same size and minimise edges cut between them.
# A graph partitioning backend such as Metis or KaHyPar needs to be installed for this function to work.
# """
# function subgraphs(
#   g::AbstractGraph; npartitions=nothing, nvertices_per_partition=nothing, kwargs...
# )
#   return subgraphs(g, subgraph_vertices(g; npartitions, nvertices_per_partition, kwargs...))
# end

"""
  TODO: do we want to make it a public function?
"""
function _noncommoninds(partition::DataGraph)
  networks = [Vector{ITensor}(partition[v]) for v in vertices(partition)]
  network = vcat(networks...)
  return noncommoninds(network...)
end

# Util functions for partition
function _commoninds(partition::DataGraph)
  networks = [Vector{ITensor}(partition[v]) for v in vertices(partition)]
  network = vcat(networks...)
  outinds = noncommoninds(network...)
  allinds = mapreduce(t -> [i for i in inds(t)], vcat, network)
  return Vector(setdiff(allinds, outinds))
end
