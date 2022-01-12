struct IndsNetwork{I,V} <: AbstractIndsNetwork{I,V}
  data_graph::UniformDataGraph{Vector{I},V}
end
data_graph(is::IndsNetwork) = getfield(is, :data_graph)
underlying_graph(is::IndsNetwork) = underlying_graph(data_graph(is))

function IndsNetwork(g::CustomVertexGraph, link_space::Nothing, site_space::Nothing)
  dg = DataGraph{Vector{Index},Vector{Index}}(g)
  return IndsNetwork(dg)
end

function IndsNetwork(g::CustomVertexGraph, link_space, site_space)
  is = IndsNetwork(g, nothing, nothing)
  for e in edges(is)
    is[e] = [Index(link_space, edge_tag(e))]
  end
  for v in vertices(is)
    is[v] = [Index(site_space, vertex_tag(v))]
  end
  return is
end

function IndsNetwork(g::CustomVertexGraph; link_space=nothing, site_space=nothing)
  return IndsNetwork(g, link_space, site_space)
end

copy(is::IndsNetwork) = IndsNetwork(copy(data_graph(is)))
