import ITensors: siteinds

struct MetaGraph{G<:AbstractGraph,E,EM,V,VM} <: AbstractGraph{V}
  parent_graph::G
  vertex_metadata::Dictionary{V,VM}
  edge_metadata::Dictionary{E,EM}
  function CustomVertexGraph{V,G}(parent_graph::AbstractGraph{V}, vertices, vertex_to_parent_vertex_map) where {L,G,T}
    return new{L,G,T}(parent_graph, vertices, vertex_to_parent_vertex_map)
  end
end

function siteinds(sitetype::String, dims::Tuple{Int64, Int64})
  return siteinds(sitetype, CustomVertexGraph(dims))
end

function siteinds(sitetype::String, graph::AbstractGraph)
  sites = siteinds(sitetype, nv(g))
  return IndsNetwork(graph, sites)
end
