using DataGraphs: DataGraphs, DataGraph
using Dictionaries: dictionary
using ITensors: ITensor
using NamedGraphs: NamedGraphs, NamedEdge, NamedGraph, vertextype

struct Private end

"""
    ITensorNetwork
"""
struct ITensorNetwork{V} <: AbstractITensorNetwork{V}
  data_graph::DataGraph{V,ITensor,ITensor,NamedGraph{V},NamedEdge{V}}
  global function _ITensorNetwork(data_graph::DataGraph)
    return new{vertextype(data_graph)}(data_graph)
  end
end

#
# Data access
#

data_graph(tn::ITensorNetwork) = getfield(tn, :data_graph)
data_graph_type(TN::Type{<:ITensorNetwork}) = fieldtype(TN, :data_graph)

function DataGraphs.underlying_graph_type(TN::Type{<:ITensorNetwork})
  return fieldtype(data_graph_type(TN), :underlying_graph)
end

# Versions taking vertex types.
function ITensorNetwork{V}() where {V}
  # TODO: Is there a better way to write this?
  return _ITensorNetwork(data_graph_type(ITensorNetwork{V})())
end
function ITensorNetwork{V}(tn::ITensorNetwork) where {V}
  # TODO: Is there a better way to write this?
  return _ITensorNetwork(DataGraph{V}(data_graph(tn)))
end
function ITensorNetwork{V}(g::NamedGraph) where {V}
  # TODO: Is there a better way to write this?
  return ITensorNetwork(NamedGraph{V}(g))
end

ITensorNetwork() = ITensorNetwork{Any}()

# Conversion
# TODO: Copy or not?
ITensorNetwork(tn::ITensorNetwork) = copy(tn)

NamedGraphs.convert_vertextype(::Type{V}, tn::ITensorNetwork{V}) where {V} = tn
NamedGraphs.convert_vertextype(V::Type, tn::ITensorNetwork) = ITensorNetwork{V}(tn)

Base.copy(tn::ITensorNetwork) = _ITensorNetwork(copy(data_graph(tn)))

#
# Construction from collections of ITensors
#

function itensors_to_itensornetwork(ts)
  g = NamedGraph(collect(eachindex(ts)))
  tn = ITensorNetwork(g)
  for v in vertices(g)
    tn[v] = ts[v]
  end
  return tn
end
function ITensorNetwork(ts::AbstractVector{ITensor})
  return itensors_to_itensornetwork(ts)
end
function ITensorNetwork(ts::AbstractDictionary{<:Any,ITensor})
  return itensors_to_itensornetwork(ts)
end
function ITensorNetwork(ts::AbstractDict{<:Any,ITensor})
  return itensors_to_itensornetwork(ts)
end
function ITensorNetwork(vs::AbstractVector, ts::AbstractVector{ITensor})
  return itensors_to_itensornetwork(Dictionary(vs, ts))
end
function ITensorNetwork(ts::AbstractVector{<:Pair{<:Any,ITensor}})
  return itensors_to_itensornetwork(dictionary(ts))
end
# TODO: Decide what this should do, maybe it should factorize?
function ITensorNetwork(t::ITensor)
  return itensors_to_itensornetwork([t])
end

#
# Construction from underyling named graph
#

function ITensorNetwork(
  eltype::Type, undef::UndefInitializer, graph::AbstractNamedGraph; kwargs...
)
  return ITensorNetwork(eltype, undef, IndsNetwork(graph; kwargs...))
end

function ITensorNetwork(f, graph::AbstractNamedGraph; kwargs...)
  return ITensorNetwork(f, IndsNetwork(graph; kwargs...))
end

function ITensorNetwork(graph::AbstractNamedGraph; kwargs...)
  return ITensorNetwork(IndsNetwork(graph; kwargs...))
end

#
# Construction from underyling simple graph
#

function ITensorNetwork(
  eltype::Type, undef::UndefInitializer, graph::AbstractSimpleGraph; kwargs...
)
  return ITensorNetwork(eltype, undef, IndsNetwork(graph; kwargs...))
end

function ITensorNetwork(f, graph::AbstractSimpleGraph; kwargs...)
  return ITensorNetwork(f, IndsNetwork(graph); kwargs...)
end

function ITensorNetwork(graph::AbstractSimpleGraph; kwargs...)
  return ITensorNetwork(IndsNetwork(graph); kwargs...)
end

#
# Construction from IndsNetwork
#

function ITensorNetwork(
  eltype::Type, undef::UndefInitializer, inds_network::IndsNetwork; kwargs...
)
  return ITensorNetwork(inds_network; kwargs...) do v
    return (inds...) -> ITensor(eltype, undef, inds...)
  end
end

function ITensorNetwork(eltype::Type, inds_network::IndsNetwork; kwargs...)
  return ITensorNetwork(inds_network; kwargs...) do v
    return (inds...) -> ITensor(eltype, inds...)
  end
end

function ITensorNetwork(undef::UndefInitializer, inds_network::IndsNetwork; kwargs...)
  return ITensorNetwork(inds_network; kwargs...) do v
    return (inds...) -> ITensor(undef, inds...)
  end
end

function ITensorNetwork(inds_network::IndsNetwork; kwargs...)
  return ITensorNetwork(inds_network; kwargs...) do v
    return (inds...) -> ITensor(inds...)
  end
end

# TODO: Handle `eltype` and `undef` through `generic_state`.
function generic_state(f, inds...)
  return f(inds...)
end
function generic_state(a::AbstractArray, inds...)
  return itensor(a, inds...)
end
function generic_state(s::AbstractString, inds...)
  tensor = state(s, inds[1])
  # TODO: Remove this and handle with `insert_missing_linkinds`.
  return contract(tensor, onehot.(vcat(inds[2:end]...) .=> 1)...)
end

# TODO: This is repeated from `ModelHamiltonians`, put into a
# single location (such as a `MakeCallable` submodule).
to_callable(value::Type) = value
to_callable(value::Function) = value
to_callable(value::AbstractDict) = Base.Fix1(getindex, value)
to_callable(value::AbstractDictionary) = Base.Fix1(getindex, value)
function to_callable(value::AbstractArray{<:Any,N}) where {N}
  return Base.Fix1(getindex, value) ∘ CartesianIndex
end
to_callable(value) = Returns(value)

function ITensorNetwork(value, inds_network::IndsNetwork; kwargs...)
  return ITensorNetwork(to_callable(value), inds_network; kwargs...)
end

function ITensorNetwork(elt::Type, f, inds_network::IndsNetwork; link_space=1, kwargs...)
  tn = ITensorNetwork(f, inds_network; kwargs...)
  for v in vertices(tn)
    tn[v] = elt.(tn[v])
  end
  return tn
end

function ITensorNetwork(
  itensor_constructor::Function, inds_network::IndsNetwork; link_space=1, kwargs...
)
  # Graphs.jl uses `zero` to create a graph of the same type
  # without any vertices or edges.
  inds_network_merge = typeof(inds_network)(
    underlying_graph(inds_network); link_space, kwargs...
  )
  inds_network = union(inds_network_merge, inds_network)
  tn = ITensorNetwork{vertextype(inds_network)}()
  for v in vertices(inds_network)
    add_vertex!(tn, v)
  end
  for e in edges(inds_network)
    add_edge!(tn, e)
  end
  for v in vertices(tn)
    siteinds = get(inds_network, v, indtype(inds_network)[])
    linkinds = [
      get(inds_network, edgetype(inds_network)(v, nv), indtype(inds_network)[]) for
      nv in neighbors(inds_network, v)
    ]
    # TODO: Come up with a better interface besides flattening the indices.
    # Maybe a namedtuple or dictionary, which indicates which indices
    # are site and links, which edges the links are associated with, etc.
    tensor_v = generic_state(itensor_constructor(v), siteinds..., vcat(linkinds...)...)
    # TODO: Call `insert_missing_linkinds` instead.
    setindex_preserve_graph!(tn, tensor_v, v)
  end
  return tn
end

# TODO: Remove this in favor of `insert_missing_internal_inds`
# or call it a different name, such as `factorize_edges`.
function insert_links(ψ::ITensorNetwork, edges::Vector=edges(ψ); cutoff=1e-15)
  for e in edges
    # Define this to work?
    # ψ = factorize(ψ, e; cutoff)
    ψᵥ₁, ψᵥ₂ = factorize(ψ[src(e)] * ψ[dst(e)], inds(ψ[src(e)]); cutoff, tags=edge_tag(e))
    ψ[src(e)] = ψᵥ₁
    ψ[dst(e)] = ψᵥ₂
  end
  return ψ
end

ITensorNetwork(itns::Vector{ITensorNetwork}) = reduce(⊗, itns)

function Base.Vector{ITensor}(ψ::ITensorNetwork)
  return ITensor[ψ[v] for v in vertices(ψ)]
end
