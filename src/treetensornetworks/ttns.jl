"""
    TreeTensorNetworkState{V} <: AbstractITensorNetwork{V}

# Fields

- itensor_network::ITensorNetwork{V}
- ortho_lims::Vector{V}: A vector of vertices defining the orthogonality limits.

"""
struct TreeTensorNetworkState{V} <: AbstractTreeTensorNetwork{V}
  itensor_network::ITensorNetwork{V}
  ortho_center::Vector{V}
  function TreeTensorNetworkState{V}(
    itensor_network::ITensorNetwork, ortho_center::Vector=vertices(itensor_network)
  ) where {V}
    @assert is_tree(itensor_network)
    return new{V}(itensor_network, ortho_center)
  end
end

function data_graph_type(G::Type{<:TreeTensorNetworkState})
  return data_graph_type(fieldtype(G, :itensor_network))
end

function copy(ψ::TreeTensorNetworkState)
  return TreeTensorNetworkState(copy(ψ.itensor_network), copy(ψ.ortho_center))
end

const TTNS = TreeTensorNetworkState

# Field access
itensor_network(ψ::TreeTensorNetworkState) = getfield(ψ, :itensor_network)

# Required for `AbstractITensorNetwork` interface
data_graph(ψ::TreeTensorNetworkState) = data_graph(itensor_network(ψ))

# 
# Constructor
# 

TreeTensorNetworkState(tn::ITensorNetwork, args...) = TreeTensorNetworkState{vertextype(tn)}(tn, args...)

# catch-all for default ElType
function (::Type{TTNT})(g::AbstractGraph, args...; kwargs...) where {TTNT<:TTNS}
  return TTNT(Float64, g, args...; kwargs...)
end

function TreeTensorNetworkState(::Type{ElT}, graph::AbstractGraph, args...; kwargs...) where {ElT<:Number}
  itensor_network = ITensorNetwork(ElT, graph; kwargs...)
  return TreeTensorNetworkState(itensor_network, args...)
end

# construct from given state (map)
function TreeTensorNetworkState(
  ::Type{ElT}, is::IndsNetwork, initstate, args...
) where {ElT<:Number}
  itensor_network = ITensorNetwork(ElT, is, initstate)
  return TreeTensorNetworkState(itensor_network, args...)
end

# TODO: randomcircuitTTNS?
function randomTTNS(args...; kwargs...)
  T = TTNS(args...; kwargs...)
  randn!.(vertex_data(T))
  normalize!.(vertex_data(T))
  return T
end

function productTTNS(args...; kwargs...)
  return TTNS(args...; link_space=1, kwargs...)
end

# 
# Utility
# 

function replacebond!(T::TTNS, edge::AbstractEdge, phi::ITensor; kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  swapsites::Bool = get(kwargs, :swapsites, false)
  which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
  normalize::Bool = get(kwargs, :normalize, false)

  indsTe = inds(T[src(edge)])
  if swapsites
    sb = siteinds(M, src(edge))
    sbp1 = siteinds(M, dst(edge))
    indsTe = replaceinds(indsTe, sb, sbp1)
  end

  L, R, spec = factorize(
    phi, indsTe; which_decomp=which_decomp, tags=tags(T, edge), kwargs...
  )

  T[src(edge)] = L
  T[dst(edge)] = R
  if ortho == "left"
    normalize && (T[dst(edge)] ./= norm(T[dst(edge)]))
    isortho(T) && (T = set_ortho_center(T, [dst(edge)]))
  elseif ortho == "right"
    normalize && (T[src(edge)] ./= norm(T[src(edge)]))
    isortho(T) && (T = set_ortho_center(T, [src(edge)]))
  end
  return spec
end

function replacebond!(T::TTNS, edge::Pair, phi::ITensor; kwargs...)
  return replacebond!(T, edgetype(T)(edge), phi; kwargs...)
end

function replacebond(T0::TTNS, args...; kwargs...)
  return replacebond!(copy(T0), args...; kwargs...)
end
