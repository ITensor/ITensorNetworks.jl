abstract type AbstractITensorNetwork <:
              AbstractNamedDimDataGraph{ITensor,ITensor,Tuple,NamedDimEdge{Tuple}} end

# Field access
data_graph(graph::AbstractITensorNetwork) = _not_implemented()

# Overload if needed
is_directed(::Type{<:AbstractITensorNetwork}) = false

# Copy
copy(tn::AbstractITensorNetwork) = _not_implemented()

# NamedDimGraph indexing
# TODO: Define for DataGraphs
# to_vertex(tn::AbstractITensorNetwork, args...) = to_vertex(data_graph(tn), args...)
to_vertex(tn::AbstractITensorNetwork, args...) = to_vertex(underlying_graph(tn), args...)

# AbstractDataGraphs overloads
function vertex_data(graph::AbstractITensorNetwork, args...)
  return vertex_data(data_graph(graph), args...)
end
edge_data(graph::AbstractITensorNetwork, args...) = edge_data(data_graph(graph), args...)

underlying_graph(tn::AbstractITensorNetwork) = underlying_graph(data_graph(tn))
function vertex_to_parent_vertex(tn::AbstractITensorNetwork)
  return vertex_to_parent_vertex(underlying_graph(tn))
end

function getindex(tn::AbstractITensorNetwork, index...)
  return getindex(IndexType(tn, index...), tn, index...)
end

function getindex(::SliceIndex, tn::AbstractITensorNetwork, index...)
  return ITensorNetwork(getindex(data_graph(tn), index...))
end

isassigned(tn::AbstractITensorNetwork, index...) = isassigned(data_graph(tn), index...)

#
# Data modification
#

function setindex_preserve_graph!(tn::AbstractITensorNetwork, value, index...)
  setindex!(data_graph(tn), value, index...)
  return tn
end

function setindex!(tn::AbstractITensorNetwork, value, index...)
  v = to_vertex(tn, index...)
  setindex_preserve_graph!(tn, value, v)
  for edge in incident_edges(tn, v)
    rem_edge!(tn, edge)
  end
  for vertex in vertices(tn)
    if v ≠ vertex
      edge = v => vertex
      if hascommoninds(tn, edge)
        add_edge!(tn, edge)
      end
    end
  end
  return tn
end

# Convert to a collection of ITensors (`Vector{ITensor}`).
function Vector{ITensor}(tn::AbstractITensorNetwork)
  return [tn[v] for v in vertices(tn)]
end

# Convenience wrapper
itensors(tn::AbstractITensorNetwork) = Vector{ITensor}(tn)

#
# Conversion to Graphs
#

function Graph(tn::AbstractITensorNetwork)
  return Graph(Vector{ITensor}(tn))
end

function NamedDimGraph(tn::AbstractITensorNetwork)
  return NamedDimGraph(Vector{ITensor}(tn))
end

#
# Conversion to IndsNetwork
#

# Convert to an IndsNetwork
function IndsNetwork(tn::AbstractITensorNetwork)
  is = IndsNetwork(underlying_graph(tn))
  for v in vertices(tn)
    is[v] = uniqueinds(tn, v)
  end
  for e in edges(tn)
    is[e] = commoninds(tn, e)
  end
  return is
end

function siteinds(tn::AbstractITensorNetwork)
  is = IndsNetwork(underlying_graph(tn))
  for v in vertices(tn)
    is[v] = uniqueinds(tn, v)
  end
  return is
end

function linkinds(tn::AbstractITensorNetwork)
  is = IndsNetwork(underlying_graph(tn))
  for e in edges(tn)
    is[e] = commoninds(tn, e)
  end
  return is
end

#
# Index access
#

function neighbor_itensors(tn::AbstractITensorNetwork, vertex...)
  return [tn[vn] for vn in neighbors(tn, vertex...)]
end

function uniqueinds(tn::AbstractITensorNetwork, vertex...)
  return uniqueinds(tn[vertex...], neighbor_itensors(tn, vertex...)...)
end

function siteinds(tn::AbstractITensorNetwork, vertex...)
  return uniqueinds(tn, vertex...)
end

function commoninds(tn::AbstractITensorNetwork, edge)
  e = NamedDimEdge(edge)
  return commoninds(tn[src(e)], tn[dst(e)])
end

function linkinds(tn::AbstractITensorNetwork, edge)
  return commoninds(tn, edge)
end

# Priming and tagging (changing Index identifiers)
function replaceinds(tn::AbstractITensorNetwork, is_is′::Pair{<:IndsNetwork,<:IndsNetwork})
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

function map_inds(f, tn::AbstractITensorNetwork, args...; kwargs...)
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
    function $f(n::Union{IndsNetwork,AbstractITensorNetwork}, args...; kwargs...)
      return map_inds($f, n, args...; kwargs...)
    end
  end
end

adjoint(tn::Union{IndsNetwork,AbstractITensorNetwork}) = prime(tn)

dag(tn::AbstractITensorNetwork) = map_vertex_data(dag, tn)

# TODO: should this make sure that internal indices
# don't clash?
function ⊗(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; kwargs...)
  return ⊔(tn1, tn2; kwargs...)
end

# TODO: name `inner_network` to denote it is lazy?
# TODO: should this make sure that internal indices
# don't clash?
function inner(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
  return dag(tn1) ⊗ tn2
end

# TODO: how to define this lazily?
#norm(tn::AbstractITensorNetwork) = sqrt(inner(tn, tn))

function contract(tn::AbstractITensorNetwork; sequence=vertices(tn), kwargs...)
  sequence_linear_index = deepmap(v -> vertex_to_parent_vertex(tn)[v], sequence)
  return contract(Vector{ITensor}(tn); sequence=sequence_linear_index, kwargs...)
end

function contract(tn::AbstractITensorNetwork, edge::Pair)
  return contract(tn, edgetype(tn)(edge))
end

function contract(tn::AbstractITensorNetwork, edge::AbstractEdge)
  tn = copy(tn)
  new_itensor = tn[src(edge)] * tn[dst(edge)]
  rem_vertex!(tn, dst(edge))
  tn[src(edge)] = new_itensor
  return tn
end

# TODO: map to the vertex names!
function optimal_contraction_sequence(tn::AbstractITensorNetwork)
  seq_linear_index = optimal_contraction_sequence(Vector{ITensor}(tn))
  return deepmap(n -> vertices(tn)[n], seq_linear_index)
end

# TODO: should this make sure that internal indices
# don't clash?
function hvncat(
  dim::Int, tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; new_dim_names=(1, 2)
)
  dg = hvncat(dim, data_graph(tn1), data_graph(tn2); new_dim_names)

  # Add in missing edges that may be shared
  # across `tn1` and `tn2`.
  vertices1 = vertices(dg)[1:nv(tn1)]
  vertices2 = vertices(dg)[(nv(tn1) + 1):end]
  for v1 in vertices1, v2 in vertices2
    if hascommoninds(dg[v1], dg[v2])
      add_edge!(dg, v1 => v2)
    end
  end

  # TODO: Allow customization of the output type.
  ## return promote_type(typeof(tn1), typeof(tn2))(dg)
  ## return contract_output(typeof(tn1), typeof(tn2))(dg)

  return ITensorNetwork(dg)
end

#
# Printing
#

function show(io::IO, mime::MIME"text/plain", graph::AbstractITensorNetwork)
  println(io, "$(typeof(graph)) with $(nv(graph)) vertices:")
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

show(io::IO, graph::AbstractITensorNetwork) = show(io, MIME"text/plain"(), graph)

function visualize(
  tn::AbstractITensorNetwork,
  args...;
  vertex_labels_prefix=nothing,
  vertex_labels=nothing,
  kwargs...,
)
  if !isnothing(vertex_labels_prefix)
    vertex_labels = [vertex_labels_prefix * string(v) for v in vertices(tn)]
  end
  return visualize(Vector{ITensor}(tn), args...; vertex_labels, kwargs...)
end
