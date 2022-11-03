NamedGraphs.NamedDimGraph(vertices::Vector) = NamedDimGraph(tuple.(vertices))
function NamedGraphs.NamedDimGraph(vertices::Vector{<:Tuple})
  return NamedDimGraph(Graph(length(vertices)); vertices)
end

function rename_vertices(e::AbstractEdge, name_map::Dictionary)
  return typeof(e)(name_map[src(e)], name_map[dst(e)])
end

function rename_vertices(g::NamedDimGraph, name_map::Dictionary)
  original_vertices = vertices(g)
  new_vertices = getindices(name_map, original_vertices)
  new_g = NamedDimGraph(new_vertices)
  for e in edges(g)
    add_edge!(new_g, rename_vertices(e, name_map))
  end
  return new_g
end

function rename_vertices(g::NamedDimGraph, name_map::Function)
  original_vertices = vertices(g)
  return rename_vertices(g, Dictionary(original_vertices, name_map.(original_vertices)))
end
