using NamedGraphs: AbstractNamedEdge

# TODO: remove once this is merged into NamedGraphs.jl

function Base.:(==)(g1::GT, g2::GT) where {GT<:AbstractNamedGraph}
  issetequal(vertices(g1), vertices(g2)) || return false
  for v in vertices(g1)
    issetequal(inneighbors(g1, v), inneighbors(g2, v)) || return false
    issetequal(outneighbors(g1, v), outneighbors(g2, v)) || return false
  end
  return true
end

# renaming routines for general named graphs

function rename_vertices(e::ET, name_map::Dictionary) where {ET<:AbstractNamedEdge}
  # strip type parameter to allow renaming to change the vertex type
  base_edge_type = Base.typename(ET).wrapper
  return base_edge_type(name_map[src(e)], name_map[dst(e)])
end

function rename_vertices(g::GT, name_map::Dictionary) where {GT<:AbstractNamedGraph}
  original_vertices = vertices(g)
  new_vertices = getindices(name_map, original_vertices)
  # strip type parameter to allow renaming to change the vertex type
  base_graph_type = Base.typename(GT).wrapper
  new_g = base_graph_type(new_vertices)
  for e in edges(g)
    add_edge!(new_g, rename_vertices(e, name_map))
  end
  return new_g
end

function rename_vertices(g::AbstractNamedGraph, name_map::Function)
  original_vertices = vertices(g)
  return rename_vertices(g, Dictionary(original_vertices, name_map.(original_vertices)))
end

function NamedGraphs.NamedGraph(vertices::Vector)
  return NamedGraph(Graph(length(vertices)), vertices)
end

function NamedGraphs.NamedDimGraph(vertices::Array)
  return NamedDimGraph(Graph(length(vertices)); vertices)
end

function NamedGraphs.NamedDimDiGraph(vertices::Array)
  return NamedDimDiGraph(DiGraph(length(vertices)); vertices)
end
