using Graphs: path_graph
using ITensors: ITensor
using NamedGraphs: vertextype

"""
    TreeTensorNetwork{V} <: AbstractTreeTensorNetwork{V}
"""
struct TreeTensorNetwork{V} <: AbstractTreeTensorNetwork{V}
  tensornetwork::ITensorNetwork{V}
  ortho_region::Vector{V}
  global function _TreeTensorNetwork(tensornetwork::ITensorNetwork, ortho_region)
    @assert is_tree(tensornetwork)
    return new{vertextype(tensornetwork)}(tensornetwork, ortho_region)
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

# Construct from dense ITensor, using IndsNetwork of site indices.
function ttn(a::ITensor, is::IndsNetwork; ortho_region=[default_root_vertex(is)], kwargs...)
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
  return random_tensornetwork(args...; kwargs...)
end

function random_mps(args...; kwargs...)
  # TODO: Check it is a path graph.
  return random_tensornetwork(args...; kwargs...)
end

function random_mps(s::Vector{<:Index}; kwargs...)
  g = path_graph(length(s))
  # TODO: Specify data type is `eltype(s)`.
  is = IndsNetwork(g)
  for v in vertices(is)
    # TODO: Allow setting with just `s[v]`.
    is[v] = [s[v]]
  end
  return random_tensornetwork(is; kwargs...)
end
