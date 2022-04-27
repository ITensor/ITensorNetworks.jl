abstract type AbstractTreeTensorNetwork <: AbstractITensorNetwork end

"""
    TreeTensorNetworkState <: AbstractITensorNetwork

# Fields

- itensor_network::ITensorNetwork
- ortho_lims::Vector{Tuple}: A vector of vertices defining the orthogonality limits.

"""
struct TreeTensorNetworkState <: AbstractTreeTensorNetwork
  itensor_network::ITensorNetwork
  ortho_center::Vector{Tuple}
  function TreeTensorNetworkState(itensor_network::ITensorNetwork, ortho_center::Vector{Tuple}=vertices(itensor_network))
    @assert is_tree(itensor_network)
    return new(itensor_network, ortho_center)
  end
end

copy(ψ::TreeTensorNetworkState) = TreeTensorNetworkState(copy(ψ.itensor_network), copy(ψ.ortho_center))

const TTNS = TreeTensorNetworkState

# Field access
ITensorNetwork(ψ::TreeTensorNetworkState) = ψ.itensor_network

# Constructor
function TreeTensorNetworkState(inds_network::IndsNetwork, args...; kwargs...)
  return TreeTensorNetworkState(ITensorNetwork(inds_network; kwargs...), args...)
end

function TreeTensorNetworkState(graph::AbstractGraph, args...; kwargs...)
  itensor_network = ITensorNetwork(graph; kwargs...)
  return TreeTensorNetworkState(itensor_network, args...)
end

# Required for `AbstractITensorNetwork` interface
data_graph(ψ::TreeTensorNetworkState) = data_graph(ITensorNetwork(ψ))
