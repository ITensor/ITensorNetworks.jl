using DataGraphs: DataGraphs, DataGraph
using Dictionaries: Indices, dictionary
using ITensors: ITensors, ITensor, op, state
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
  # Try using `convert_vertextype`.
  new_data_graph_type = data_graph_type(ITensorNetwork{V})
  new_underlying_graph_type = underlying_graph_type(new_data_graph_type)
  return _ITensorNetwork(new_data_graph_type(new_underlying_graph_type()))
end
function ITensorNetwork{V}(tn::ITensorNetwork) where {V}
  # TODO: Is there a better way to write this?
  # Try using `convert_vertextype`.
  return _ITensorNetwork(DataGraph{V}(data_graph(tn)))
end
function ITensorNetwork{V}(g::NamedGraph) where {V}
  # TODO: Is there a better way to write this?
  # Try using `convert_vertextype`.
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

function ITensorNetwork(eltype::Type, undef::UndefInitializer, is::IndsNetwork; kwargs...)
  return ITensorNetwork(is; kwargs...) do v
    return (inds...) -> ITensor(eltype, undef, inds...)
  end
end

function ITensorNetwork(eltype::Type, is::IndsNetwork; kwargs...)
  return ITensorNetwork(is; kwargs...) do v
    return (inds...) -> ITensor(eltype, inds...)
  end
end

function ITensorNetwork(undef::UndefInitializer, is::IndsNetwork; kwargs...)
  return ITensorNetwork(is; kwargs...) do v
    return (inds...) -> ITensor(undef, inds...)
  end
end

function ITensorNetwork(is::IndsNetwork; kwargs...)
  return ITensorNetwork(is; kwargs...) do v
    return (inds...) -> ITensor(inds...)
  end
end

# TODO: Handle `eltype` and `undef` through `generic_state`.
# `inds` are stored in a `NamedTuple`
function generic_state(f, inds::NamedTuple)
  return generic_state(f, reduce(vcat, inds.linkinds; init=inds.siteinds))
end

function generic_state(f, inds::Vector)
  return f(inds)
end
function generic_state(a::AbstractArray, inds::Vector)
  return itensor(a, inds)
end
function generic_state(x::Op, inds::NamedTuple)
  # TODO: Figure out what to do if there is more than one site.
  if !isempty(inds.siteinds)
    @assert length(inds.siteinds) == 2
    i = inds.siteinds[findfirst(i -> plev(i) == 0, inds.siteinds)]
    @assert i' ∈ inds.siteinds
    site_tensors = [op(x.which_op, i)]
  else
    site_tensors = []
  end
  link_tensors = [[onehot(i => 1) for i in inds.linkinds[e]] for e in keys(inds.linkinds)]
  return contract(reduce(vcat, link_tensors; init=site_tensors))
end
function generic_state(s::AbstractString, inds::NamedTuple)
  # TODO: Figure out what to do if there is more than one site.
  site_tensors = [state(s, only(inds.siteinds))]
  link_tensors = [[onehot(i => 1) for i in inds.linkinds[e]] for e in keys(inds.linkinds)]
  return contract(reduce(vcat, link_tensors; init=site_tensors))
end

# TODO: This is similar to `ModelHamiltonians.to_callable`,
# try merging the two.
to_callable(value::Type) = value
to_callable(value::Function) = value
function to_callable(value::AbstractDict)
  return Base.Fix1(getindex, value) ∘ keytype(value)
end
function to_callable(value::AbstractDictionary)
  return Base.Fix1(getindex, value) ∘ keytype(value)
end
function to_callable(value::AbstractArray{<:Any,N}) where {N}
  return Base.Fix1(getindex, value) ∘ CartesianIndex
end
to_callable(value) = Returns(value)

function ITensorNetwork(value, is::IndsNetwork; kwargs...)
  return ITensorNetwork(to_callable(value), is; kwargs...)
end

function ITensorNetwork(
  elt::Type, f, is::IndsNetwork; link_space=trivial_space(is), kwargs...
)
  tn = ITensorNetwork(f, is; kwargs...)
  for v in vertices(tn)
    # TODO: Ideally we would use broadcasting, i.e. `elt.(tn[v])`,
    # but that doesn't work right now on ITensors.
    tn[v] = ITensors.convert_eltype(elt, tn[v])
  end
  return tn
end

function ITensorNetwork(
  itensor_constructor::Function, is::IndsNetwork; link_space=trivial_space(is), kwargs...
)
  is = insert_linkinds(is; link_space)
  tn = ITensorNetwork{vertextype(is)}()
  for v in vertices(is)
    add_vertex!(tn, v)
  end
  for e in edges(is)
    add_edge!(tn, e)
  end
  for v in vertices(tn)
    # TODO: Replace with `is[v]` once `getindex(::IndsNetwork, ...)` is smarter.
    siteinds = get(is, v, Index[])
    edges = [edgetype(is)(v, nv) for nv in neighbors(is, v)]
    linkinds = map(e -> is[e], Indices(edges))
    tensor_v = generic_state(itensor_constructor(v), (; siteinds, linkinds))
    setindex_preserve_graph!(tn, tensor_v, v)
  end
  return tn
end

ITensorNetwork(itns::Vector{ITensorNetwork}) = reduce(⊗, itns)

# TODO: Use `vertex_data` here?
function eachtensor(ψ::ITensorNetwork)
  # This type declaration is needed to narrow
  # the element type of the resulting `Dictionary`,
  # raise and issue with `Dictionaries.jl`.
  return map(v -> ψ[v]::ITensor, vertices(ψ))
end
