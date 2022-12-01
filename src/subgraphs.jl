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
Given a graph g, form 'nv(g) ÷ nvertices_per_partition' subgraphs. Try to keep all subgraphs the same size and minimise edges cut between them
Returns a datagraph where each vertex contains the list of vertices involved in that subgraph. The edges state which subgraphs are connected
KaHyPar needs to be installed for this function to work
"""
function subgraphs(g::AbstractGraph, nvertices_per_partition::Integer; kwargs...)
  @show nvertices_per_partition
  @show nv(g)
  nvertices_per_partition = min(nv(g), nvertices_per_partition)
  @show nvertices_per_partition
  npartitions = nv(g) ÷ nvertices_per_partition
  vertex_to_partition = partition(g, npartitions; kwargs...)
  partition_vertices = groupfind(vertex_to_partition)
  if length(partition_vertices) ≠ npartitions
    @warn "Requested $nvertices_per_partition vertices per partition for a graph with $(nv(g)) vertices. Attempting to partition the graph into $npartitions partitions but instead it was partitioned into $(length(partition_vertices)) partitions."
  end
  @assert issetequal(keys(partition_vertices), 1:length(partition_vertices))
  dg_subgraphs = DataGraph(NamedGraph(npartitions), groupfind(vertex_to_partition))
  for e in edges(g)
    s1 = findfirst_on_vertices(subgraph_vertices -> src(e) ∈ subgraph_vertices, dg_subgraphs)
    s2 = findfirst_on_vertices(subgraph_vertices -> dst(e) ∈ subgraph_vertices, dg_subgraphs)
    if (!has_edge(dg_subgraphs, s1, s2) && s1 != s2)
      add_edge!(dg_subgraphs, s1, s2)
    end
  end
  return dg_subgraphs
end

## """
## Given a graph g on a d-dimensional grid of size Ls[1] x Ls[2] x ..., form subgraphs of size ls[1] x ls[2] in a regular fashion
## Return the subgraphs (1... npartitions) with their contained vertices. Also return a dictionary of the subgraphs connected to each sgraph
## """
## function subgraphs_grid(g::NamedGraph, Ls::Vector{Int64}, ls::Vector{Int64})
##   lengths = Ls ./ ls
##   ps = Dict{Tuple,Int64}()
##   for v in vertices(g)
##     pos = []
##     count = 1
##     for i in 1:length(v)
##       push!(pos, ceil(Int64, v[i] / ls[i]))
##       if (pos[i] == 1)
##         p = 1
##       else
##         p = prod(lengths[1:(i - 1)])
##       end
##       count = Int(count + (pos[i] - 1) * p)
##     end
##     ps[v] = count
##   end
## 
##   nsubgraphs = Int(prod(lengths))
## 
##   dg_subgraphs = DataGraph{Vector{Tuple},Any}(NamedGraph([(i,) for i in 1:nsubgraphs]))
##   for s in 1:nsubgraphs
##     dg_subgraphs[(s,)] = [v for v in vertices(g) if ps[v] == s]
##   end
## 
##   for e in edges(g)
##     v1, v2 = src(e), dst(e)
##     s1, s2 = find_subgraph(v1, dg_subgraphs), find_subgraph(v2, dg_subgraphs)
##     if (!has_edge(dg_subgraphs, s1, s2) && s1 != s2)
##       add_edge!(dg_subgraphs, s1, s2)
##     end
##   end
## 
##   return dg_subgraphs
## end
