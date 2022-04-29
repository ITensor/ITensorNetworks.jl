abstract type AbstractTreeTensorNetwork <: AbstractITensorNetwork end

function default_root_vertex(ϕ::AbstractTreeTensorNetwork, ψ::AbstractTreeTensorNetwork)
  return first(vertices(ψ))
end

function inner(
  ϕ::AbstractTreeTensorNetwork,
  ψ::AbstractTreeTensorNetwork;
  root_vertex=default_root_vertex(ϕ, ψ),
)
  ϕᴴ = sim(dag(ψ); sites=[])
  ψ = sim(ψ; sites=[])
  ϕψ = ϕᴴ ⊗ ψ
  # TODO: find the largest tensor and use it as
  # the `root_vertex`.
  root_vertex = first(vertices(ψ))
  for e in post_order_dfs_edges(ψ, root_vertex)
    if has_vertex(ϕψ, 2, src(e)...)
      ϕψ = contract(ϕψ, (2, src(e)...) => (1, src(e)...))
    end
    ϕψ = contract(ϕψ, (1, src(e)...) => (1, dst(e)...))
    if has_vertex(ϕψ, 2, dst(e)...)
      ϕψ = contract(ϕψ, (2, dst(e)...) => (1, dst(e)...))
    end
  end
  return ϕψ[1, root_vertex...][]
end

function norm(ψ::AbstractTreeTensorNetwork)
  return √(abs(real(inner(ψ, ψ))))
end

function orthogonalize(ψ::AbstractTreeTensorNetwork, root_vertex...)
  for e in post_order_dfs_edges(ψ, root_vertex)
    ψ = orthogonalize(ψ, e)
  end
  return ψ
end

# For ambiguity error
function orthogonalize(tn::AbstractTreeTensorNetwork, edge::AbstractEdge; kwargs...)
  return _orthogonalize_edge(tn, edge; kwargs...)
end

"""
    TreeTensorNetworkState <: AbstractITensorNetwork

# Fields

- itensor_network::ITensorNetwork
- ortho_lims::Vector{Tuple}: A vector of vertices defining the orthogonality limits.

"""
struct TreeTensorNetworkState <: AbstractTreeTensorNetwork
  itensor_network::ITensorNetwork
  ortho_center::Vector{Tuple}
  function TreeTensorNetworkState(
    itensor_network::ITensorNetwork, ortho_center::Vector{Tuple}=vertices(itensor_network)
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
