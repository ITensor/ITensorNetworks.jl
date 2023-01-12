## """
##     TreeTensorNetwork <: AbstractITensorNetwork
## 
## A finite size tree tensor network operator type. 
## Keeps track of the orthogonality center.
## 
## # Fields
## 
## - itensor_network::ITensorNetwork{V}
## - ortho_lims::Vector{V}: A vector of vertices defining the orthogonality limits.
## """
## struct TreeTensorNetwork{V} <: AbstractTreeTensorNetwork{V}
##   itensor_network::ITensorNetwork{V}
##   ortho_center::Vector{V}
##   function TreeTensorNetwork{V}(
##     itensor_network::ITensorNetwork, ortho_center::Vector=vertices(itensor_network)
##   ) where {V}
##     @assert is_tree(itensor_network)
##     return new{V}(itensor_network, ortho_center)
##   end
## end

## function data_graph_type(G::Type{<:TreeTensorNetwork})
##   return data_graph_type(fieldtype(G, :itensor_network))
## end

## function copy(ψ::TreeTensorNetwork)
##   return TreeTensorNetwork(copy(ψ.itensor_network), copy(ψ.ortho_center))
## end

## const TTN = TreeTensorNetwork

## # Field access
## itensor_network(ψ::TreeTensorNetwork) = getfield(ψ, :itensor_network)

## # Required for `AbstractITensorNetwork` interface
## data_graph(ψ::TreeTensorNetwork) = data_graph(itensor_network(ψ))

# 
# Constructor
# 

## TreeTensorNetwork(tn::ITensorNetwork, args...) = TreeTensorNetwork{vertextype(tn)}(tn, args...)

## # catch-all for default ElType
## function (::Type{TTNT})(g::AbstractGraph, args...; kwargs...) where {TTNT<:TTN}
##   return TTNT(Float64, g, args...; kwargs...)
## end

## function TreeTensorNetwork(::Type{ElT}, graph::AbstractGraph, args...; kwargs...) where {ElT<:Number}
##   itensor_network = ITensorNetwork(ElT, graph; kwargs...)
##   return TreeTensorNetwork(itensor_network, args...)
## end

# 
# Inner products
# 

## # TODO: implement using multi-graph disjoint union
## function inner(y::AbstractTTN, A::AbstractTTN, x::AbstractTTN; root_vertex=default_root_vertex(x, A, y))
##   traversal_order = reverse(post_order_dfs_vertices(x, root_vertex))
##   check_hascommoninds(siteinds, A, x)
##   check_hascommoninds(siteinds, A, y)
##   ydag = sim(dag(y); sites=[])
##   x = sim(x; sites=[])
##   O = ydag[root_vertex] * A[root_vertex] * x[root_vertex]
##   for v in traversal_order[2:end]
##     O = O * ydag[v] * A[v] * x[v]
##   end
##   return O[]
## end

## # TODO: implement using multi-graph disjoint
## function inner(
##   B::AbstractTTN, y::AbstractTTN, A::AbstractTTN, x::AbstractTTN; root_vertex=default_root_vertex(B, y, A, x)
## )
##   N = nv(B)
##   if nv(y) != N || nv(x) != N || nv(A) != N
##     throw(
##       DimensionMismatch(
##         "inner: mismatched number of vertices $N and $(nv(x)) or $(nv(y)) or $(nv(A))"
##       ),
##     )
##   end
##   check_hascommoninds(siteinds, A, x)
##   check_hascommoninds(siteinds, B, y)
##   for v in vertices(B)
##     !hascommoninds(
##       uniqueinds(siteinds(A, v), siteinds(x, v)), uniqueinds(siteinds(B, v), siteinds(y, v))
##     ) && error(
##       "$(typeof(x)) Ax and $(typeof(y)) By must share site indices. On site $v, Ax has site indices $(uniqueinds(siteinds(A, v), (siteinds(x, v)))) while By has site indices $(uniqueinds(siteinds(B, v), siteinds(y, v))).",
##     )
##   end
##   ydag = sim(linkinds, dag(y))
##   Bdag = sim(linkinds, dag(B))
##   traversal_order = reverse(post_order_dfs_vertices(x, root_vertex))
##   yB = ydag[root_vertex] * Bdag[root_vertex]
##   Ax = A[root_vertex] * x[root_vertex]
##   O = yB * Ax
##   for v in traversal_order[2:end]
##     yB = ydag[v] * Bdag[v]
##     Ax = A[v] * x[v]
##     yB *= O
##     O = yB * Ax
##   end
##   return O[]
## end

## # 
## # Construction from operator (map)
## # 
## 
## function TTN(
##   ::Type{ElT}, sites::IndsNetwork, ops::Dictionary; kwargs...
## ) where {ElT<:Number}
##   N = nv(sites)
##   os = Prod{Op}()
##   for v in vertices(sites)
##     os *= Op(ops[v], v)
##   end
##   T = TTN(ElT, os, sites; kwargs...)
##   # see https://github.com/ITensor/ITensors.jl/issues/526
##   lognormT = lognorm(T)
##   T /= exp(lognormT / N) # TODO: fix broadcasting for in-place assignment
##   truncate!(T; cutoff=1e-15)
##   T *= exp(lognormT / N)
##   return T
## end
## 
## function TTN(
##   ::Type{ElT}, sites::IndsNetwork, fops::Function; kwargs...
## ) where {ElT<:Number}
##   ops = Dictionary(vertices(sites), map(v -> fops(v), vertices(sites)))
##   return TTN(ElT, sites, ops; kwargs...)
## end
## 
## function TTN(::Type{ElT}, sites::IndsNetwork, op::String; kwargs...) where {ElT<:Number}
##   ops = Dictionary(vertices(sites), fill(op, nv(sites)))
##   return TTN(ElT, sites, ops; kwargs...)
## end

# 
# Conversion
# 

## function convert(::Type{<:TTN}, T::TTN)
##   return TTN(itensor_network(T), ortho_center(T))
## end

## function convert(::Type{<:TTN}, T::TTN)
##   return TTN(itensor_network(T), ortho_center(T))
## end
