struct IndsNetwork{I,V} <: AbstractIndsNetwork{I,V}
  data_graph::UniformDataGraph{Vector{I},V}
end
data_graph(is::IndsNetwork) = getfield(is, :data_graph)
underlying_graph(is::IndsNetwork) = underlying_graph(data_graph(is))

#
# Visualization
#

function visualize(is::IndsNetwork, args...; kwargs...)
  return visualize(ITensorNetwork(is), args...; kwargs...)
end

function IndsNetwork(g::NamedDimGraph, link_space::Nothing, site_space::Nothing)
  dg = DataGraph{Vector{Index},Vector{Index}}(g)
  return IndsNetwork(dg)
end

function IndsNetwork(g::NamedDimGraph, link_space, site_space)
  is = IndsNetwork(g, nothing, nothing)
  if !isnothing(link_space)
    for e in edges(is)
      is[e] = [Index(link_space, edge_tag(e))]
    end
  end
  if !isnothing(site_space)
    for v in vertices(is)
      is[v] = [Index(site_space, vertex_tag(v))]
    end
  end
  return is
end

# TODO: Maybe make `link_space` and `site_space` functions of the edges and vertices.
function IndsNetwork(g::NamedDimGraph; link_space=nothing, site_space=nothing)
  return IndsNetwork(g, link_space, site_space)
end

copy(is::IndsNetwork) = IndsNetwork(copy(data_graph(is)))

function map_inds(f, is::IndsNetwork, args...; sites=nothing, links=nothing, kwargs...)
  return map_data(i -> f(i, args...; kwargs...), is; vertices=sites, edges=links)
end
