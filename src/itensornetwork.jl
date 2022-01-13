struct ITensorNetwork{V} <: AbstractITensorNetwork{V}
  data_graph::UniformDataGraph{ITensor,V}
end

#
# Data access
#

data_graph(tn::ITensorNetwork) = getfield(tn, :data_graph)
underlying_graph(tn::ITensorNetwork) = underlying_graph(data_graph(tn))

getindex(tn::ITensorNetwork, I1, I2, I...) = getindex(tn, (I1, I2, I...))
isassigned(tn::ITensorNetwork, I1, I2, I...) = isassigned(tn, (I1, I2, I...))

#
# Data modification
#

# TODO: Make a version of `setindex!` that doesn't preserve the graph (recomputes the `underlying_graph` as needed).
function setindex_preserve_graph!(tn::ITensorNetwork, edge_or_vertex, data)
  setindex!(data_graph(tn), edge_or_vertex, data)
  return tn
end

setindex!(tn::ITensorNetwork, x, I1, I2, I...) = setindex!(tn, x, (I1, I2, I...))

copy(tn::ITensorNetwork) = ITensorNetwork(copy(data_graph(tn)))

#
# Construction from Graphs
#

# TODO: Add sitespace, linkspace
function ITensorNetwork(g::Union{Graph,CustomVertexGraph})
  dg = DataGraph{ITensor,ITensor}(g)
  return ITensorNetwork(dg)
end

#
# Construction from IndsNetwork
#

# Assigns indices with space `link_space` to unassigned
# edges of the network.
function _ITensorNetwork(is::IndsNetwork, link_space)
  edge_data(e) = [edge_index(e, link_space)]
  is_assigned = assign_data(is; edge_data)
  return _ITensorNetwork(is_assigned, nothing)
end

function _ITensorNetwork(is::IndsNetwork, link_space::Nothing)
  g = underlying_graph(is)
  tn = ITensorNetwork(g)
  for v in vertices(tn)
    siteinds = is[v]
    linkinds = [is[v => nv] for nv in neighbors(is, v)]
    tn[v] = ITensor(siteinds, linkinds...)
  end
  return tn
end

function ITensorNetwork(is::IndsNetwork; link_space=nothing)
  return _ITensorNetwork(is, link_space)
end

# Convert to a collection of ITensors (`Vector{ITensor}`).
function itensors(tn::ITensorNetwork)
  return collect(vertex_data(tn))
end

#
# Conversion to IndsNetwork
#

function incident_edges(g::AbstractGraph{V}, v::V) where {V}
  return [edgetype(g)(v, vn) for vn in neighbors(g, v)]
end

# Convert to an IndsNetwork
function IndsNetwork(tn::ITensorNetwork)
  is = IndsNetwork(underlying_graph(tn))
  for v in vertices(tn)
    is[v] = uniqueinds(tn, v)
  end
  for e in edges(tn)
    is[e] = commoninds(tn, e)
  end
  return is
end

function siteinds(tn::ITensorNetwork)
  is = IndsNetwork(underlying_graph(tn))
  for v in vertices(tn)
    is[v] = uniqueinds(tn, v)
  end
  return is
end

function linkinds(tn::ITensorNetwork)
  is = IndsNetwork(underlying_graph(tn))
  for e in edges(tn)
    is[e] = commoninds(tn, e)
  end
  return is
end

#
# Index access
#

function neighbor_itensors(tn::ITensorNetwork{V}, v::V) where {V}
  return [tn[vn] for vn in neighbors(tn, v)]
end

function uniqueinds(tn::ITensorNetwork{V}, v::V) where {V}
  return uniqueinds(tn[v], neighbor_itensors(tn, v)...)
end

function siteinds(tn::ITensorNetwork{V}, v::V) where {V}
  return uniqueinds(tn, v)
end

function commoninds(tn::ITensorNetwork{V}, e::CustomVertexEdge{V}) where {V}
  return commoninds(tn[src(e)], tn[dst(e)])
end

function linkinds(tn::ITensorNetwork{V}, e::CustomVertexEdge{V}) where {V}
  return commoninds(tn, e)
end

function linkinds(tn::ITensorNetwork, e)
  return linkinds(tn, edgetype(tn)(e))
end

# Priming and tagging (changing Index identifiers)
function replaceinds(tn::ITensorNetwork, is_is′::Pair{<:IndsNetwork,<:IndsNetwork})
  tn = copy(tn)
  is, is′ = is_is′
  # TODO: Check that `is` and `is′` have the same vertices and edges.
  for v in vertices(is)
    setindex_preserve_graph!(tn, replaceinds(tn[v], is[v] => is′[v]), v)
  end
  for e in edges(is)
    for v in (src(e), dst(e))
      setindex_preserve_graph!(tn, replaceinds(tn[v], is[e] => is′[e]), v)
    end
  end
  return tn
end

function map_vertex_data(f, is::IndsNetwork; vertices=nothing)
  is′ = copy(is)
  vs = isnothing(vertices) ? Graphs.vertices(is) : vertices
  for v in vs
    is′[v] = f(is[v])
  end
  return is′
end

function map_edge_data(f, is::IndsNetwork; edges=nothing)
  is′ = copy(is)
  es = isnothing(edges) ? Graphs.edges(is) : edges
  for e in es
    is′[e] = f(is[e])
  end
  return is′
end

function map_data(f, is::IndsNetwork; vertices, edges)
  is = map_vertex_data(f, is; vertices)
  return map_edge_data(f, is; edges)
end

function map_inds(f, is::IndsNetwork, args...; sites=nothing, links=nothing, kwargs...)
  return map_data(i -> f(i, args...; kwargs...), is; vertices=sites, edges=links)
end

function map_inds(f, tn::ITensorNetwork, args...; kwargs...)
  is = IndsNetwork(tn)
  is′ = map_inds(f, is, args...; kwargs...)
  return replaceinds(tn, is => is′)
end

const map_inds_label_functions = [
  :prime,
  :setprime,
  :noprime,
  :replaceprime,
  :swapprime,
  :addtags,
  :removetags,
  :replacetags,
  :settags,
  :sim,
  :swaptags,
  # :replaceind,
  # :replaceinds,
  # :swapind,
  # :swapinds,
]

for f in map_inds_label_functions
  @eval begin
    function $f(n::Union{IndsNetwork,ITensorNetwork}, args...; kwargs...)
      return map_inds($f, n, args...; kwargs...)
    end
  end
end
