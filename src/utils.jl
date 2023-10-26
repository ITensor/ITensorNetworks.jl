to_tuple(x) = (x,)
to_tuple(x::Tuple) = x

function cartesian_to_linear(dims::Tuple)
  return Dictionary(vec(Tuple.(CartesianIndices(dims))), 1:prod(dims))
end

# Convert to real if possible
maybe_real(x::Real) = x
maybe_real(x::Complex) = iszero(imag(x)) ? real(x) : x

maybe_only(x) = x
maybe_only(x::Tuple{T}) where {T} = only(x)

front(itr, n=1) = Iterators.take(itr, length(itr) - n)
tail(itr) = Iterators.drop(itr, 1)

# Tree utils
function line_to_tree(line::Vector)
  if length(line) == 1 && line[1] isa Vector
    return line[1]
  end
  if length(line) <= 2
    return line
  end
  return [line_to_tree(line[1:(end - 1)]), line[end]]
end

function edge_update_order(g)
  forests = NamedGraphs.build_forest_cover(g)
  edges = NamedEdge[]
  for forest in forests
    trees = NamedGraph[forest[vs] for vs in connected_components(forest)]
    for tree in trees
      push!(edges, tree_edge_update_order(tree)...)
    end
  end

  return edges
end

#Find an optimal ordering of the edges in a tree
function tree_edge_update_order(
  g::AbstractNamedGraph; root_vertex=first(keys(sort(eccentricities(g); rev=true)))
)
  @assert is_tree(g)
  es = post_order_dfs_edges(g, root_vertex)
  return vcat(es, reverse(reverse.(es)))
end
