# TODO: Move to `Graphs.GraphsExtensions`.
using Graphs: AbstractGraph, IsDirected, a_star
using NamedGraphs.GraphsExtensions: child_vertices, is_leaf, undirected_graph, parent_vertex
using SimpleTraits: SimpleTraits, Not, @traitfn

"""Determine if an edge involves a leaf (at src or dst)"""
function is_leaf_edge(g::AbstractGraph, e)
  return is_leaf(g, src(e)) || is_leaf(g, dst(e))
end

"""Determine if a node has any neighbors which are leaves"""
function has_leaf_neighbor(g::AbstractGraph, v)
  for vn in neighbors(g, v)
    if (is_leaf(g, vn))
      return true
    end
  end
  return false
end

"""Get all edges which do not involve a leaf"""
function internal_edges(g::AbstractGraph)
  return filter(e -> !is_leaf_edge(g, e), edges(g))
end

"""Get distance of a vertex from a leaf"""
function distance_to_leaf(g::AbstractGraph, v)
  leaves = leaf_vertices(g)
  if (isempty(leaves))
    println("ERROR: GRAPH DOES NTO CONTAIN LEAVES")
    return NaN
  end

  return minimum([length(a_star(g, v, leaf)) for leaf in leaves])
end

"""Return all vertices which are within a certain pathlength `dist` of the leaves of the  graph"""
function distance_from_roots(g::AbstractGraph, dist::Int64)
  return vertices(g)[findall(<=(dist), [distance_to_leaf(g, v) for v in vertices(g)])]
end

"""
Return the root vertex of a rooted directed graph
"""
@traitfn function _root(graph::AbstractGraph::IsDirected)
  @assert _is_rooted(graph) "the input $(graph) has to be rooted"
  v = first(vertices(graph))
  while parent_vertex(graph, v) != nothing
    v = parent_vertex(graph, v)
  end
  return v
end

@traitfn function _is_rooted(graph::AbstractGraph::IsDirected)
  return isone(length(filter(v -> isnothing(parent_vertex(graph, v)), vertices(graph))))
end

@traitfn function _is_rooted_directed_binary_tree(graph::AbstractGraph::IsDirected)
  if !_is_rooted(graph)
    return false
  end
  if !is_tree(undirected_graph(graph))
    return false
  end
  for v in vertices(graph)
    if length(child_vertices(graph, v)) > 2
      return false
    end
  end
  return true
end
