using ITensors
using ITensorNetworks

using ITensorNetworks: contract_approx

using Graphs
using Dictionaries

using NamedGraphs
using NamedGraphs: AbstractNamedGraph, parent_graph, vertex_to_parent_vertex
function Graphs.degree(graph::AbstractNamedGraph, vertex)
  return degree(parent_graph(graph), vertex_to_parent_vertex(graph, vertex))
end
function Graphs.dijkstra_shortest_paths(graph::AbstractNamedGraph, vertex, distmx::AbstractMatrix)
  return dijkstra_shortest_paths(parent_graph(graph), vertex_to_parent_vertex(graph, vertex), distmx)
end
function Graphs.closeness_centrality(graph::AbstractNamedGraph, distmx::AbstractMatrix; normalize::Bool=true)
  return Dictionary(vertices(graph), closeness_centrality(parent_graph(graph), distmx; normalize))
end

using DataGraphs
using DataGraphs: underlying_graph
function Graphs.degree(graph::AbstractDataGraph, vertex)
  return degree(underlying_graph(graph), vertex)
end
function Graphs.dijkstra_shortest_paths(graph::AbstractDataGraph, vertex, distmx::AbstractMatrix)
  return dijkstra_shortest_paths(underlying_graph(graph), vertex, distmx)
end
function Graphs.closeness_centrality(graph::AbstractDataGraph, distmx::AbstractMatrix; normalize::Bool=true)
  return closeness_centrality(underlying_graph(graph), distmx; normalize)
end

g = named_grid((3, 3))
tn = randomITensorNetwork(g; link_space=2)

centrality_tn = closeness_centrality(tn)
@show argmax(centrality_tn)
@show argmin(centrality_tn)

# source_vertex = argmin(centrality_tn)
# spanning_tree = dfs_tree(tn, source_vertex)
# contraction_vertices = reverse(bfs_vertices(spanning_tree, source_vertex))
# display(contraction_vertices)
# sequence = reduce((v1, v2) -> [v1, v2], contraction_vertices)

source_vertex = argmin(centrality_tn)
spanning_tree = dfs_tree(tn, source_vertex)
contraction_vertices = reverse(post_order_dfs_vertices(spanning_tree, source_vertex))
display(contraction_vertices)
sequence = reduce((v1, v2) -> [v1, v2], contraction_vertices)

res = contract_approx(tn; sequence, maxdim=10, cutoff=1e-10)[]

