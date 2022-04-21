struct ITensorNetwork <: AbstractITensorNetwork
  data_graph::UniformDataGraph{ITensor}
end

#
# Data access
#

data_graph(tn::ITensorNetwork) = getfield(tn, :data_graph)
underlying_graph(tn::ITensorNetwork) = underlying_graph(data_graph(tn))

function getindex(tn::ITensorNetwork, index...)
  return getindex(IndexType(tn, index...), tn, index...)
end

function getindex(::SliceIndex, tn::ITensorNetwork, index...)
  return ITensorNetwork(getindex(data_graph(tn), index...))
end

# getindex(tn::ITensorNetwork, I1, I2, I...) = getindex(data_graph(tn), I1, I2, I...)
isassigned(tn::ITensorNetwork, index...) = isassigned(data_graph(tn), index...)

#
# Data modification
#

# TODO: Make a version of `setindex!` that doesn't preserve the graph (recomputes the `underlying_graph` as needed).
function setindex_preserve_graph!(tn::ITensorNetwork, edge_or_vertex, data)
  setindex!(data_graph(tn), edge_or_vertex, data)
  return tn
end

setindex!(tn::ITensorNetwork, x, I1, I2, I...) = setindex!(data_graph(tn), x, I1, I2, I...)

copy(tn::ITensorNetwork) = ITensorNetwork(copy(data_graph(tn)))

#
# Construction from collections of ITensors
#

function ITensorNetwork(ts::Vector{ITensor})
  g = NamedDimGraph(ts)
  tn = ITensorNetwork(g)
  for v in eachindex(ts)
    tn[v] = ts[v]
  end
  return tn
end

#
# Construction from Graphs
#

function _ITensorNetwork(g::NamedDimGraph, site_space::Nothing, link_space::Nothing)
  dg = NamedDimDataGraph{ITensor,ITensor}(g)
  return ITensorNetwork(dg)
end

function ITensorNetwork(g::NamedDimGraph; kwargs...)
  return ITensorNetwork(IndsNetwork(g; kwargs...))
end

function ITensorNetwork(g::Graph; kwargs...)
  return ITensorNetwork(IndsNetwork(g; kwargs...))
end

#
# Conversion to Graphs
#

function Graph(tn::ITensorNetwork)
  return Graph(Vector{ITensor}(tn))
end

function NamedDimGraph(tn::ITensorNetwork)
  return NamedDimGraph(Vector{ITensor}(tn))
end

#
# Construction from IndsNetwork
#

# Alternative implementation:
# edge_data(e) = [edge_index(e, link_space)]
# is_assigned = assign_data(is; edge_data)
function _ITensorNetwork(is::IndsNetwork, link_space)
  is_assigned = copy(is)
  for e in edges(is)
    is_assigned[e] = [edge_index(e, link_space)]
  end
  return _ITensorNetwork(is_assigned, nothing)
end

get_assigned(d, i, default) = isassigned(d, i) ? d[i] : default

function _ITensorNetwork(is::IndsNetwork, link_space::Nothing)
  g = underlying_graph(is)
  tn = _ITensorNetwork(g, nothing, nothing)
  for v in vertices(tn)
    siteinds = get_assigned(is, v, Index[])
    linkinds = [get_assigned(is, v => nv, Index[]) for nv in neighbors(is, v)]
    tn[v] = ITensor(siteinds, linkinds...)
  end
  return tn
end

function ITensorNetwork(is::IndsNetwork; link_space=nothing)
  return _ITensorNetwork(is, link_space)
end

# Convert to a collection of ITensors (`Vector{ITensor}`).
function Vector{ITensor}(tn::ITensorNetwork)
  return collect(vertex_data(tn))
end

# Convenience wrapper
itensors(tn::ITensorNetwork) = Vector{ITensor}(tn)

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

function neighbor_itensors(tn::ITensorNetwork, v::Tuple)
  return [tn[vn] for vn in neighbors(tn, v)]
end

function uniqueinds(tn::ITensorNetwork, v::Tuple)
  return uniqueinds(tn[v], neighbor_itensors(tn, v)...)
end

function siteinds(tn::ITensorNetwork, v::Tuple)
  return uniqueinds(tn, v)
end

function commoninds(tn::ITensorNetwork, e::NamedDimEdge)
  return commoninds(tn[src(e)], tn[dst(e)])
end

function linkinds(tn::ITensorNetwork, e::NamedDimEdge)
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

adjoint(tn::Union{IndsNetwork,ITensorNetwork}) = prime(tn)

dag(tn::ITensorNetwork) = map_vertex_data(dag, tn)

# TODO: replace vertices
# Dictionary(getindices(k, eachindex(x)), getindices(x, eachindex(x)))

# TODO: use vertices from the original graphs, currently
# it flattens to linear vertex labels.
# TODO: rename `tensor_product_network`, `otimes_network`,
# `ITensorNetwork`, `contract_network`, etc. to denote that it is lazy?
function contract_network(tn1::ITensorNetwork, tn2::ITensorNetwork)
  tn1 = sim(tn1; sites=[])
  tn2 = sim(tn2; sites=[])
  tns = [Vector{ITensor}(tn1); Vector{ITensor}(tn2)]
  # TODO: Define `blockdiag` for NamedDimGraph to merge the graphs,
  # then add edges to the results graph.
  # Also, automatically merge vertices. For example, reproduce
  # things like `vcat` for cases of linear indices and cartesian indices,
  # or add `Sub(1)`, `Sub(2)`, etc.
  #g1 = underlying_graph(tn1)
  #g2 = underlying_graph(tn2)
  #g = blockdiag(g1, g2)
  #add_edge!(g, commoninds(tn1, tn2))
  return ITensorNetwork(tns)
end

const ⊗ = contract_network

# TODO: name `inner_network` to denote it is lazy?
function inner(tn1::ITensorNetwork, tn2::ITensorNetwork)
  return dag(tn1) ⊗ tn2
end

# TODO: how to define this lazily?
#norm(tn::ITensorNetwork) = sqrt(inner(tn, tn))

function contract(tn::ITensorNetwork; kwargs...)
  return contract(Vector{ITensor}(tn); kwargs...)
end

function optimal_contraction_sequence(tn::ITensorNetwork)
  return optimal_contraction_sequence(Vector{ITensor}(tn))
end

#
# Printing
#

function show(io::IO, mime::MIME"text/plain", graph::ITensorNetwork)
  println(io, "ITensorNetwork with $(nv(graph)) vertices:")
  show(io, mime, vertices(graph))
  println(io, "\n")
  println(io, "and $(ne(graph)) edge(s):")
  for e in edges(graph)
    show(io, mime, e)
    println(io)
  end
  println(io)
  println(io, "with vertex data:")
  show(io, mime, inds.(vertex_data(graph)))
  return nothing
end

show(io::IO, graph::ITensorNetwork) = show(io, MIME"text/plain"(), graph)

function visualize(tn::ITensorNetwork, args...; kwargs...)
  return visualize(Vector{ITensor}(tn), args...; kwargs...)
end
