using Graphs.SimpleTraits

# TODO: remove once this is merged into NamedGraphs.jl

# paths for tree graphs

@traitfn function vertex_path(graph::::(!IsDirected), s, t)
  dfs_tree_graph = dfs_tree(graph, t...)
  return vertex_path(dfs_tree_graph, s, t)
end

@traitfn function edge_path(graph::::(!IsDirected), s, t)
  dfs_tree_graph = dfs_tree(graph, t...)
  return edge_path(dfs_tree_graph, s, t)
end

# assumes the graph is a rooted directed tree with root d
@traitfn function vertex_path(graph::::IsDirected, s, t)
  vertices = eltype(graph)[s]
  while vertices[end] != t
    push!(vertices, parent_vertex(graph, vertices[end]...))
  end
  return vertices
end

@traitfn function edge_path(graph::::IsDirected, s, t)
  vertices = vertex_path(graph, s, t)
  pop!(vertices)
  return [edgetype(graph)(vertex, parent_vertex(graph, vertex...)) for vertex in vertices]
end
