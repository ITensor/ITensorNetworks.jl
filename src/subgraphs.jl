function findall_on_vertices(f::Function, graph::AbstractDataGraph)
  return findall(f, vertex_data(graph))
end

function findfirst_on_vertices(f::Function, graph::AbstractDataGraph)
  return findfirst(f, vertex_data(graph))
end

function findall_on_edges(f::Function, graph::AbstractDataGraph)
  return findall(f, edge_data(graph))
end

function findfirst_on_edges(f::Function, graph::AbstractDataGraph)
  return findfirst(f, edge_data(graph))
end

"""
Find the subgraph which contains the specified vertex.

TODO: Rename something more general, like:

findfirst_in_vertex_data(item, graph::AbstractDataGraph)
"""
function find_subgraph(vertex, subgraphs::DataGraph)
  return findfirst_on_vertices(subgraph_vertices -> vertex ∈ subgraph_vertices, subgraphs)
end


"""
Take an abstract graph and a grouping of the vertices (specified by a dict: group_name -> vertices in group) and create a datagraph of the subgraphs.
Edges connect subgraphs which contain vertices connected in g. Edge data is the number of edges in g between subgraphs
"""
function create_subgraphs(g::AbstractGraph, vertex_groups::Dictionary)
  dg_subgraph = DataGraph{vertextype(NamedGraph(keys(vertex_groups))), Vector{Any}, Int64}(NamedGraph(keys(vertex_groups)), vertex_groups)
  for e in edges(g)
    s1 = findfirst_on_vertices(vertex_groups -> src(e) ∈ vertex_groups, dg_subgraph)
    s2 = findfirst_on_vertices(vertex_groups -> dst(e) ∈ vertex_groups, dg_subgraph)
    if (s1 != s2)
      if(!has_edge(dg_subgraph, s1, s2))
          add_edge!(dg_subgraph, s1, s2)
          dg_subgraph[s1=>s2] = 1
      else
          dg_subgraph[s1=>s2] += 1
      end
    end

  end
  return dg_subgraph
end

"""
Graph partitioning functions. One of Metis or KaHyPar needs to be installed for these to work
"""

"""create a partitioning of the vertices of g, nvertices per partition
Returns a dictionary of partition_no -> vertices in partition
"""
function partition_vertices(g::AbstractGraph, nvertices_per_partition::Integer; kwargs...)
  nvertices_per_partition = min(nv(g), nvertices_per_partition)
  npartitions = nv(g) ÷ nvertices_per_partition
  vertex_to_partition = partition(g, npartitions; kwargs...)
  partitioned_vertices = groupfind(vertex_to_partition)
  if length(partitioned_vertices) ≠ npartitions
    @warn "Requested $nvertices_per_partition vertices per partition for a graph with $(nv(g)) vertices. Attempting to partition the graph into $npartitions partitions but instead it was partitioned into $(length(partition_vertices)) partitions. Partitions are $partition_vertices."
  end
  if !issetequal(keys(partitioned_vertices), 1:length(partitioned_vertices))
    @warn "Vertex partioning is $partitioned_vertices, which may not be what you were hoping for. If not, try a different graph partitioning algorithm or backend."
  end
  return partitioned_vertices
end

"""create a partitioning of the vertices of g, npartitions
Returns a dictionary of partition_no -> vertices in partition
"""
function partition_vertices(g::AbstractGraph; npartitions=nv(g), kwargs...)
  nvertices_per_partition = floor(Int, nv(g)/npartitions)
  return partition_vertices(g, nvertices_per_partition; kwargs...)
end

"""
Form subgraphs (returns a datagraph where vertices of g are grouped into larger vertices of the datagraph. 
Aims to have nvertices_per_subgraph grouped into each of the datagraph vertices)
"""
function create_subgraphs(g::AbstractGraph, nvertices_per_subgraph::Integer; kwargs...)
  vertex_groups = partition_vertices(g, nvertices_per_subgraph; kwargs...)
  return create_subgraphs(g, vertex_groups)
end

"""
Form subgraphs (returns a datagraph where vertices of g are grouped into larger vertices of the datagraph. 
Aims to have number of vertices in the datagraph be nsubgraphs)
"""
function create_subgraphs(g::AbstractGraph; nsubgraphs = 1, kwargs...)
  vertex_groups = partition_vertices(g; nsubgraphs = nsubgraphs, kwargs...)
  return create_subgraphs(g, vertex_groups)
end