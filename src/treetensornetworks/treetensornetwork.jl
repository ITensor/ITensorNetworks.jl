using Graphs: path_graph
using ITensors: ITensor
using LinearAlgebra: factorize, normalize
using NamedGraphs.GraphsExtensions: GraphsExtensions, vertextype

"""
    TreeTensorNetwork{V} <: AbstractTreeTensorNetwork{V}
"""
struct TreeTensorNetwork{V,OrthoRegion} <: AbstractTreeTensorNetwork{V}
  tensornetwork::ITensorNetwork{V}
  ortho_region::OrthoRegion
  global function _TreeTensorNetwork(tensornetwork::ITensorNetwork, ortho_region)
    @assert is_tree(tensornetwork)
    @assert vertextype(tensornetwork) === eltype(ortho_region)
    return new{vertextype(tensornetwork),typeof(ortho_region)}(tensornetwork, ortho_region)
  end
  global function _TreeTensorNetwork(tensornetwork::ITensorNetwork)
    return _TreeTensorNetwork(tensornetwork, vertices(tensornetwork))
  end
end

function TreeTensorNetwork(tn::ITensorNetwork; ortho_region=vertices(tn))
  return _TreeTensorNetwork(tn, ortho_region)
end
function TreeTensorNetwork{V}(tn::ITensorNetwork) where {V}
  return TreeTensorNetwork(ITensorNetwork{V}(tn))
end
function TreeTensorNetwork{V,OrthoRegion}(tn::ITensorNetwork) where {V,OrthoRegion}
  return TreeTensorNetwork(ITensorNetwork{V}(tn); ortho_region=OrthoRegion(vertices(tn)))
end

const TTN = TreeTensorNetwork

# Field access
ITensorNetwork(tn::TTN) = getfield(tn, :tensornetwork)
ortho_region(tn::TTN) = getfield(tn, :ortho_region)

# Required for `AbstractITensorNetwork` interface
data_graph(tn::TTN) = data_graph(ITensorNetwork(tn))

function data_graph_type(G::Type{<:TTN})
  return data_graph_type(fieldtype(G, :tensornetwork))
end

function Base.copy(tn::TTN)
  return _TreeTensorNetwork(copy(ITensorNetwork(tn)), copy(ortho_region(tn)))
end

# 
# Constructor
# 

function set_ortho_region(tn::TTN, ortho_region)
  return ttn(ITensorNetwork(tn); ortho_region)
end

function ttn(args...; ortho_region=nothing)
  tn = ITensorNetwork(args...)
  if isnothing(ortho_region)
    ortho_region = vertices(tn)
  end
  return _TreeTensorNetwork(tn, ortho_region)
end

function mps(args...; ortho_region=nothing)
  # TODO: Check it is a path graph.
  tn = ITensorNetwork(args...)
  if isnothing(ortho_region)
    ortho_region = vertices(tn)
  end
  return _TreeTensorNetwork(tn, ortho_region)
end

function mps(f, is::Vector{<:Index}; kwargs...)
  return mps(f, path_indsnetwork(is); kwargs...)
end

# Construct from dense ITensor, using IndsNetwork of site indices.
function ttn(
  a::ITensor,
  is::IndsNetwork;
  ortho_region=[GraphsExtensions.default_root_vertex(is)],
  kwargs...,
)
  for v in vertices(is)
    @assert hasinds(a, is[v])
  end
  @assert ortho_region ⊆ vertices(is)
  tn = ITensorNetwork(is)
  ortho_center = first(ortho_region)
  for e in post_order_dfs_edges(tn, ortho_center)
    left_inds = uniqueinds(is, e)
    a_l, a_r = factorize(a, left_inds; tags=edge_tag(e), ortho="left", kwargs...)
    tn[src(e)] = a_l
    is[e] = commoninds(a_l, a_r)
    a = a_r
  end
  tn[ortho_center] = a
  ttn_a = ttn(tn)
  return orthogonalize(ttn_a, ortho_center)
end

function random_ttn(args...; kwargs...)
  # TODO: Check it is a tree graph.
  return normalize(_TreeTensorNetwork(random_tensornetwork(args...; kwargs...)))
end

function random_mps(args...; kwargs...)
  # TODO: Check it is a path graph.
  return random_ttn(args...; kwargs...)
end

function random_mps(f, is::Vector{<:Index}; kwargs...)
  return random_mps(f, path_indsnetwork(is); kwargs...)
end

function random_mps(s::Vector{<:Index}; kwargs...)
  return random_mps(path_indsnetwork(s); kwargs...)
end
