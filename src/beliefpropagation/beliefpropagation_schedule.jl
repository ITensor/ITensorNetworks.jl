abstract type EdgeSequenceAlgorithm end

struct ForestCover <: EdgeSequenceAlgorithm end
struct Parallel <: EdgeSequenceAlgorithm end

default_edge_sequence_alg() = ForestCover()

default_bp_niters(g::NamedGraph) = is_tree(g) ? 1 : nothing
function default_bp_niters(g::Union{DataGraph,AbstractITensorNetwork})
  return default_bp_niters(undirected_graph(underlying_graph(g)))
end

function edge_sequence(g::NamedGraph; alg=default_edge_sequence_alg())
  return edge_sequence(alg, g)
end

function edge_sequence(mts::DataGraph; alg=default_edge_sequence_alg())
  return edge_sequence(alg, undirected_graph(underlying_graph(mts)))
end

function edge_sequence(
  ::ForestCover, g::NamedGraph; root_vertex=NamedGraphs.default_root_vertex
)
  @assert !is_directed(g)
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

function edge_sequence(
  ::ForestCover, mts::DataGraph; root_vertex=NamedGraphs.default_root_vertex
)
  return edge_sequence(ForestCover, undirected_graph(underlying_graph(mts)); root_vertex)
end

function edge_sequence(::Parallel, g::NamedGraph)
  @assert !is_directed(g)
  return [[e] for e in vcat(edges(g), reverse.(edges(g)))]
end

function edge_sequence(::Parallel, mts::DataGraph)
  return edge_sequence(Parallel, undirected_graph(underlying_graph(mts)))
end
