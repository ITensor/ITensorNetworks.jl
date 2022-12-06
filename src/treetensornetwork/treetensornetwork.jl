# TODO: Replace `AbstractITensorNetwork` with a trait `IsTree`.
abstract type AbstractTreeTensorNetwork{V} <: AbstractITensorNetwork{V} end

underlying_graph_type(G::Type{<:AbstractTreeTensorNetwork}) = underlying_graph_type(data_graph_type(G))

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
    if has_vertex(ϕψ, (src(e), 2))
      ϕψ = contract(ϕψ, (src(e), 2) => (src(e), 1))
    end
    ϕψ = contract(ϕψ, (src(e), 1) => (dst(e), 1))
    if has_vertex(ϕψ, (dst(e), 2))
      ϕψ = contract(ϕψ, (dst(e), 2) => (dst(e), 1))
    end
  end
  return ϕψ[root_vertex, 1][]
end

function norm(ψ::AbstractTreeTensorNetwork)
  return √(abs(real(inner(ψ, ψ))))
end

function orthogonalize(ψ::AbstractTreeTensorNetwork, root_vertex)
  for e in post_order_dfs_edges(ψ, root_vertex)
    ψ = orthogonalize(ψ, e)
  end
  return ψ
end

# For ambiguity error
function orthogonalize(tn::AbstractTreeTensorNetwork, edge::AbstractEdge; kwargs...)
  return typeof(tn)(orthogonalize(ITensorNetwork(tn), edge; kwargs...))
end

function orthogonalize(tn::AbstractTreeTensorNetwork, edge::Pair; kwargs...)
  return orthogonalize(tn, edgetype(tn)(edge); kwargs...)
end

"""
    TreeTensorNetworkState <: AbstractITensorNetwork

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

data_graph_type(G::Type{<:TreeTensorNetworkState}) = data_graph_type(fieldtype(G, :itensor_network))

function copy(ψ::TreeTensorNetworkState)
  return TreeTensorNetworkState(copy(ψ.itensor_network), copy(ψ.ortho_center))
end

const TTNS = TreeTensorNetworkState

# Field access
itensor_network(ψ::TreeTensorNetworkState) = getfield(ψ, :itensor_network)

# Constructor
TreeTensorNetworkState(tn::ITensorNetwork, args...) = TreeTensorNetworkState{vertextype(tn)}(tn, args...)

function TreeTensorNetworkState(inds_network::IndsNetwork, args...; kwargs...)
  return TreeTensorNetworkState(ITensorNetwork(inds_network; kwargs...), args...)
end

function TreeTensorNetworkState(graph::AbstractGraph, args...; kwargs...)
  itensor_network = ITensorNetwork(graph; kwargs...)
  return TreeTensorNetworkState(itensor_network, args...)
end

# Required for `AbstractITensorNetwork` interface
data_graph(ψ::TreeTensorNetworkState) = data_graph(itensor_network(ψ))
