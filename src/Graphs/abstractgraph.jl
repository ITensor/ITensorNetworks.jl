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

"""Get all vertices which are leaves of a graph"""
function leaf_vertices(g::AbstractGraph)
  return vertices(g)[findall(==(1), [is_leaf(g, v) for v in vertices(g)])]
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
