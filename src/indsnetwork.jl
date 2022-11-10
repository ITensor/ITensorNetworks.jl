struct IndsNetwork{I} <: AbstractIndsNetwork{I}
  data_graph::UniformDataGraph{Vector{I}}
end
data_graph(is::IndsNetwork) = getfield(is, :data_graph)
underlying_graph(is::IndsNetwork) = underlying_graph(data_graph(is))

#
# Constructor
#

function IndsNetwork(g::NamedDimGraph, link_space::Nothing, site_space::Nothing)
  dg = NamedDimDataGraph{Vector{Index},Vector{Index}}(g)
  return IndsNetwork(dg)
end

function IndsNetwork(g::Graph, args...; dims=nothing, vertices=nothing, kwargs...)
  return IndsNetwork(NamedDimGraph(g; dims, vertices), args...; kwargs...)
end

# union type for objects used to specify an index when calling `ITensors.Index`
const IndexSpec = Union{Integer,ITensors.QNBlocks}
# union type for objects that can represent a tensor index
const IndexLike = Union{Index,IndexSpec}

_maybe_index(i::IndexSpec, v::Tuple) = Index(i, vertex_tag(v))
_maybe_index(i::IndexSpec, e::NamedDimEdge) = Index(i, edge_tag(e))
_maybe_index(i::Index, v) = i

_maybe_vector(x) = [x]
_maybe_vector(x::Vector{<:IndexLike}) = x

function IndsNetwork(
  g::NamedDimGraph,
  link_space::AbstractDictionary{NamedDimEdge{Tuple}},
  site_space::AbstractDictionary{Tuple},
)
  is = IndsNetwork(g, nothing, nothing)
  for e in edges(is)
    if !isnothing(link_space[e])
      is[e] = _maybe_index.(_maybe_vector(link_space[e]), Ref(e))
    end
  end
  for v in vertices(is)
    if !isnothing(site_space[v])
      is[v] = _maybe_index.(_maybe_vector(site_space[v]), Ref(v))
    end
  end
  return is
end

function IndsNetwork(g::NamedDimGraph, link_space, site_space::AbstractDictionary{Tuple})
  # convert link_space to Dictionary of edges
  link_space_map = Dictionary(edges(g), fill(link_space, ne(g)))
  return IndsNetwork(g, link_space_map, site_space)
end

function IndsNetwork(g::NamedDimGraph, link_space, site_space)
  # convert site_space to Dictionary of vertices
  site_space_map = Dictionary(vertices(g), fill(site_space, nv(g)))
  return IndsNetwork(g, link_space, site_space_map)
end

"""
  IndsNetwork(g::NamedDimGraph; link_space=nothing, site_spaces=nothing)

Construct an `IndsNetwork` by supplying a graph and specifying the `link_space` and
`site_space` of the network.

Both `link_space` and `site_space` can be supplied as any of the following:
- `nothing`
- An `Integer`, `Vector{Pair{ITensors.QN, Int64}}`, or a `Vector` of either of these
  specifying the set of indices uniformly for all links/sites.
- A `Dictionary` mapping each vertex (edge) to an `Index`, `Integer`,
  `Vector{Pair{ITensors.QN, Int64}}` or a `Vector` of either of these specifying the set of
  indices on each site (link) individually.
"""
function IndsNetwork(g::NamedDimGraph; link_space=nothing, site_space=nothing)
  return IndsNetwork(g, link_space, site_space)
end

#
# Utility
#

copy(is::IndsNetwork) = IndsNetwork(copy(data_graph(is)))

function map_inds(f, is::IndsNetwork, args...; sites=nothing, links=nothing, kwargs...)
  return map_data(i -> f(i, args...; kwargs...), is; vertices=sites, edges=links)
end

#
# Visualization
#

function visualize(is::IndsNetwork, args...; kwargs...)
  return visualize(ITensorNetwork(is), args...; kwargs...)
end
