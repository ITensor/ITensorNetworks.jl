"""
    TreeTensorNetworkOperator <: AbstractITensorNetwork

A finite size tree tensor network operator type. 
Keeps track of the orthogonality center.

# Fields

- itensor_network::ITensorNetwork
- ortho_lims::Vector{Tuple}: A vector of vertices defining the orthogonality limits.
"""
mutable struct TreeTensorNetworkOperator <: AbstractTreeTensorNetwork
  itensor_network::ITensorNetwork
  ortho_center::Vector{Tuple}
  function TreeTensorNetworkOperator(
    itensor_network::ITensorNetwork, ortho_center::Vector{<:Tuple}=vertices(itensor_network)
  )
    @assert is_tree(itensor_network)
    return new(itensor_network, ortho_center)
  end
end

function copy(ψ::TreeTensorNetworkOperator)
  return TreeTensorNetworkOperator(copy(ψ.itensor_network), copy(ψ.ortho_center))
end

const TTNO = TreeTensorNetworkOperator

# Field access
ITensorNetwork(ψ::TreeTensorNetworkOperator) = ψ.itensor_network

# Required for `AbstractITensorNetwork` interface
data_graph(ψ::TreeTensorNetworkOperator) = data_graph(ITensorNetwork(ψ))

# 
# Constructor
# 

# catch-all for default ElType
function TreeTensorNetworkOperator(graph::AbstractGraph, args...; kwargs...)
  return TreeTensorNetworkOperator(Float64, graph, args...; kwargs...)
end

function TreeTensorNetworkOperator(
  ::Type{ElT}, graph::AbstractGraph, args...; kwargs...
) where {ElT<:Number}
  itensor_network = ITensorNetwork(ElT, graph; kwargs...)
  return TreeTensorNetworkOperator(itensor_network, args...)
end

# 
# Inner products
# 

# TODO: implement using multi-graph disjoint union
function inner(y::TTNS, A::TTNO, x::TTNS; root_vertex=default_root_vertex(x, A, y))
  traversal_order = reverse(post_order_dfs_vertices(x, root_vertex))
  check_hascommoninds(siteinds, A, x)
  check_hascommoninds(siteinds, A, y)
  ydag = sim(dag(y); sites=[])
  x = sim(x; sites=[])
  O = ydag[root_vertex] * A[root_vertex] * x[root_vertex]
  for v in traversal_order[2:end]
    O = O * ydag[v] * A[v] * x[v]
  end
  return O[]
end

# TODO: implement using multi-graph disjoint
function inner(
  B::TTNO, y::TTNS, A::TTNO, x::TTNS; root_vertex=default_root_vertex(B, y, A, x)
)
  N = nv(B)
  if nv(y) != N || nv(x) != N || nv(A) != N
    throw(
      DimensionMismatch(
        "inner: mismatched number of vertices $N and $(nv(x)) or $(nv(y)) or $(nv(A))"
      ),
    )
  end
  check_hascommoninds(siteinds, A, x)
  check_hascommoninds(siteinds, B, y)
  for v in vertices(B)
    !hascommoninds(
      uniqueinds(siteinds(A, v), siteinds(x, v)), uniqueinds(siteinds(B, v), siteinds(y, v))
    ) && error(
      "$(typeof(x)) Ax and $(typeof(y)) By must share site indices. On site $v, Ax has site indices $(uniqueinds(siteinds(A, v), (siteinds(x, v)))) while By has site indices $(uniqueinds(siteinds(B, v), siteinds(y, v))).",
    )
  end
  ydag = sim(linkinds, dag(y))
  Bdag = sim(linkinds, dag(B))
  traversal_order = reverse(post_order_dfs_vertices(x, root_vertex))
  yB = ydag[root_vertex] * Bdag[root_vertex]
  Ax = A[root_vertex] * x[root_vertex]
  O = yB * Ax
  for v in traversal_order[2:end]
    yB = ydag[v] * Bdag[v]
    Ax = A[v] * x[v]
    yB *= O
    O = yB * Ax
  end
  return O[]
end

# 
# Construction from operator (map)
# 

function TTNO(
  ::Type{ElT}, sites::IndsNetwork, ops::Dictionary; kwargs...
) where {ElT<:Number}
  N = nv(sites)
  os = Prod{Op}()
  for v in vertices(sites)
    os *= Op(ops[v], v)
  end
  T = TTNO(ElT, os, sites; kwargs...)
  # see https://github.com/ITensor/ITensors.jl/issues/526
  lognormT = lognorm(T)
  T /= exp(lognormT / N) # TODO: fix broadcasting for in-place assignment
  truncate!(T; cutoff=1e-15)
  T *= exp(lognormT / N)
  return T
end

function TTNO(
  ::Type{ElT}, sites::IndsNetwork, fops::Function; kwargs...
) where {ElT<:Number}
  ops = Dictionary(vertices(sites), map(v -> fops(v), vertices(sites)))
  return TTNO(ElT, sites, ops; kwargs...)
end

function TTNO(::Type{ElT}, sites::IndsNetwork, op::String; kwargs...) where {ElT<:Number}
  ops = Dictionary(vertices(sites), fill(op, nv(sites)))
  return TTNO(ElT, sites, ops; kwargs...)
end

# 
# Conversion
# 

function convert(::Type{TTNS}, T::TTNO)
  return TTNS(ITensorNetwork(T), ortho_center(T))
end

function convert(::Type{TTNO}, T::TTNS)
  return TTNO(ITensorNetwork(T), ortho_center(T))
end
