"""
    TreeTensorNetworkState <: AbstractITensorNetwork

# Fields

- itensor_network::ITensorNetwork
- ortho_lims::Vector{Tuple}: A vector of vertices defining the orthogonality limits.

"""
mutable struct TreeTensorNetworkState <: AbstractTreeTensorNetwork
  itensor_network::ITensorNetwork
  ortho_center::Vector{Tuple}
  function TreeTensorNetworkState(
    itensor_network::ITensorNetwork, ortho_center::Vector{<:Tuple}=vertices(itensor_network)
  )
    @assert is_tree(itensor_network)
    return new(itensor_network, ortho_center)
  end
end

function copy(ψ::TreeTensorNetworkState)
  return TreeTensorNetworkState(copy(ψ.itensor_network), copy(ψ.ortho_center))
end

const TTNS = TreeTensorNetworkState

# Field access
ITensorNetwork(ψ::TreeTensorNetworkState) = ψ.itensor_network

# Required for `AbstractITensorNetwork` interface
data_graph(ψ::TreeTensorNetworkState) = data_graph(ITensorNetwork(ψ))

# 
# Constructor
# 

# catch-all for default ElType
function TreeTensorNetworkState(g::AbstractGraph, args...; kwargs...)
  return TreeTensorNetworkState(Float64, g, args...; kwargs...)
end

# can defer almost everything to ITensorNework constructor
function TreeTensorNetworkState(
  ::Type{ElT}, graph::AbstractGraph, args...; kwargs...
) where {ElT<:Number}
  itensor_network = ITensorNetwork(ElT, graph; kwargs...)
  return TreeTensorNetworkState(itensor_network, args...)
end

# construct from given state (map)
function TreeTensorNetworkState(
  ::Type{ElT}, is::IndsNetwork, states, args...
) where {ElT<:Number}
  itensor_network = ITensorNetwork(ElT, is, states)
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
    isortho(T) && set_ortho_center!(T, [dst(edge)])
  elseif ortho == "right"
    normalize && (T[src(edge)] ./= norm(T[src(edge)]))
    isortho(T) && set_ortho_center!(T, [src(edge)])
  end
  return spec
end

function replacebond!(T::TTNS, edge::Pair, phi::ITensor; kwargs...)
  return replacebond!(T, edgetype(T)(edge), phi; kwargs...)
end

function replacebond(T0::TTNS, args...; kwargs...)
  return replacebond!(copy(T0), args...; kwargs...)
end

# 
# Expectation values
# 

# TODO: temporary patch, to be implemented properly
function expect(psi::TTNS, opname::String; kwargs...)
  s = siteinds(psi)
  sites = get(kwargs, :sites, vertices(psi))
  res = Dictionary(sites, Vector{ComplexF64}(undef, length(sites)))
  norm2_psi = inner(psi, psi)
  for v in sites
    Opsi = copy(psi)
    Opsi[v] *= op(opname, s[v])
    noprime!(Opsi[v])
    res[v] = inner(psi, Opsi) / norm2_psi
  end
  return res
end
