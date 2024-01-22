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
  __roots = _roots(graph)
  @assert length(__roots) == 1 "the input $(graph) has to be rooted"
  return __roots[1]
end

@traitfn function _roots(graph::AbstractGraph::IsDirected)
  return [v for v in vertices(graph) if parent_vertex(graph, v) == nothing]
end

@traitfn function _is_rooted(graph::AbstractGraph::IsDirected)
  return length(_roots(graph)) == 1
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
