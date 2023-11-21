default_edge_sequence_alg() = "forest_cover"

@traitfn default_bp_niters(g::::(!IsDirected)) = is_tree(g) ? 1 : nothing
@traitfn function default_bp_niters(g::::IsDirected)
  return default_bp_niters(undirected_graph(underlying_graph(g)))
end

@traitfn function edge_sequence(
  g::::(!IsDirected); alg=default_edge_sequence_alg(), kwargs...
)
  return edge_sequence(Algorithm(alg), g; kwargs...)
end

@traitfn function edge_sequence(g::::IsDirected; alg=default_edge_sequence_alg(), kwargs...)
  return edge_sequence(Algorithm(alg), undirected_graph(underlying_graph(g)); kwargs...)
end

@traitfn function edge_sequence(alg::Algorithm, g::::IsDirected; kwargs...)
  return edge_sequence(alg, undirected_graph(underlying_graph(g)); kwargs...)
end

@traitfn function edge_sequence(
  ::Algorithm"forest_cover", g::::(!IsDirected); root_vertex=NamedGraphs.default_root_vertex
)
  forests = NamedGraphs.forest_cover(g)
  edges = edgetype(g)[]
  for forest in forests
    trees = [forest[vs] for vs in connected_components(forest)]
    for tree in trees
      tree_edges = post_order_dfs_edges(tree, root_vertex(tree))
      push!(edges, vcat(tree_edges, reverse(reverse.(tree_edges)))...)
    end
  end

  return edges
end

@traitfn function edge_sequence(::Algorithm"parallel", g::::(!IsDirected))
  return [[e] for e in vcat(edges(g), reverse.(edges(g)))]
end
