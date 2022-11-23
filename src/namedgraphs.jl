# TODO: Move to NamedGraphs.jl
# NamedGraphs.NamedGraph(vertices::Vector) = NamedGraph(tuple.(vertices))
# function NamedGraphs.NamedGraph(vertices::Vector{<:Tuple})
#   return NamedGraph(Graph(length(vertices)); vertices)
# end

# TODO: This is already defined in NamedGraphs.jl
# function rename_vertices(e::AbstractEdge, name_map::Dictionary)
#   return typeof(e)(name_map[src(e)], name_map[dst(e)])
# end
# 
# function rename_vertices(g::NamedGraph, name_map::Dictionary)
#   original_vertices = vertices(g)
#   new_vertices = getindices(name_map, original_vertices)
#   new_g = NamedGraph(new_vertices)
#   for e in edges(g)
#     add_edge!(new_g, rename_vertices(e, name_map))
#   end
#   return new_g
# end
# 
# function rename_vertices(g::NamedGraph, name_map::Function)
#   original_vertices = vertices(g)
#   return rename_vertices(g, Dictionary(original_vertices, name_map.(original_vertices)))
# end
