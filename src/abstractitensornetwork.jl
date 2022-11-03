abstract type AbstractITensorNetwork <:
              AbstractNamedDimDataGraph{ITensor,ITensor,Tuple,NamedDimEdge{Tuple}} end

# Field access
data_graph(graph::AbstractITensorNetwork) = _not_implemented()

# Overload if needed
is_directed(::Type{<:AbstractITensorNetwork}) = false

# Copy
copy(tn::AbstractITensorNetwork) = _not_implemented()

# NamedDimGraph indexing
# TODO: Define for DataGraphs
# to_vertex(tn::AbstractITensorNetwork, args...) = to_vertex(data_graph(tn), args...)
to_vertex(tn::AbstractITensorNetwork, args...) = to_vertex(underlying_graph(tn), args...)

# AbstractDataGraphs overloads
function vertex_data(graph::AbstractITensorNetwork, args...)
  return vertex_data(data_graph(graph), args...)
end
edge_data(graph::AbstractITensorNetwork, args...) = edge_data(data_graph(graph), args...)

underlying_graph(tn::AbstractITensorNetwork) = underlying_graph(data_graph(tn))
function vertex_to_parent_vertex(tn::AbstractITensorNetwork)
  return vertex_to_parent_vertex(underlying_graph(tn))
end

function getindex(tn::AbstractITensorNetwork, index...)
  return getindex(IndexType(tn, index...), tn, index...)
end

function getindex(::SliceIndex, tn::AbstractITensorNetwork, index...)
  return ITensorNetwork(getindex(data_graph(tn), index...))
end

isassigned(tn::AbstractITensorNetwork, index...) = isassigned(data_graph(tn), index...)

#
# Iteration
#

# TODO: discuss if this is the desired behavior
Base.eachindex(tn::AbstractITensorNetwork) = vertices(tn)
Base.iterate(tn::AbstractITensorNetwork) = iterate(vertex_data(tn))
Base.iterate(tn::AbstractITensorNetwork, state) = iterate(vertex_data(tn), state)

# TODO: different `map` functionalities as defined for ITensors.AbstractMPS

# TODO: broadcasting

#
# Data modification
#

function setindex_preserve_graph!(tn::AbstractITensorNetwork, value, index...)
  setindex!(data_graph(tn), value, index...)
  return tn
end

function hascommoninds(tn::AbstractITensorNetwork, edge::Pair)
  return hascommoninds(tn, edgetype(tn)(edge))
end

function hascommoninds(tn::AbstractITensorNetwork, edge::AbstractEdge)
  return hascommoninds(tn[src(edge)], tn[dst(edge)])
end

function setindex!(tn::AbstractITensorNetwork, value, index...)
  v = to_vertex(tn, index...)
  setindex_preserve_graph!(tn, value, v)
  for edge in incident_edges(tn, v)
    rem_edge!(tn, edge)
  end
  for vertex in vertices(tn)
    if v ≠ vertex
      edge = v => vertex
      if hascommoninds(tn, edge)
        add_edge!(tn, edge)
      end
    end
  end
  return tn
end

# Convert to a collection of ITensors (`Vector{ITensor}`).
function Vector{ITensor}(tn::AbstractITensorNetwork)
  return [tn[v] for v in vertices(tn)]
end

# Convenience wrapper
itensors(tn::AbstractITensorNetwork) = Vector{ITensor}(tn)

#
# Promotion and conversion
#

function LinearAlgebra.promote_leaf_eltypes(tn::AbstractITensorNetwork)
  return LinearAlgebra.promote_leaf_eltypes(itensors(tn))
end

function ITensors.promote_itensor_eltype(tn::AbstractITensorNetwork)
  return LinearAlgebra.promote_leaf_eltypes(tn)
end

ITensors.scalartype(tn::AbstractITensorNetwork) = LinearAlgebra.promote_leaf_eltypes(tn)

# TODO: eltype(::AbstractITensorNetwork) (cannot behave the same as eltype(::ITensors.AbstractMPS))

# TODO: mimic ITensors.AbstractMPS implementation using map
function ITensors.convert_leaf_eltype(eltype::Type, tn::AbstractITensorNetwork)
  tn = copy(tn)
  vertex_data(tn) .= ITensors.convert_leaf_eltype.(Ref(eltype), vertex_data(tn))
  return tn
end

# TODO: mimic ITensors.AbstractMPS implementation using map
function NDTensors.convert_scalartype(eltype::Type{<:Number}, tn::AbstractITensorNetwork)
  tn = copy(tn)
  vertex_data(tn) .= ITensors.adapt.(Ref(eltype), vertex_data(tn))
  return tn
end

#
# Conversion to Graphs
#

function Graph(tn::AbstractITensorNetwork)
  return Graph(Vector{ITensor}(tn))
end

function NamedDimGraph(tn::AbstractITensorNetwork)
  return NamedDimGraph(Vector{ITensor}(tn))
end

#
# Conversion to IndsNetwork
#

# Convert to an IndsNetwork
function IndsNetwork(tn::AbstractITensorNetwork)
  is = IndsNetwork(underlying_graph(tn))
  for v in vertices(tn)
    is[v] = uniqueinds(tn, v)
  end
  for e in edges(tn)
    is[e] = commoninds(tn, e)
  end
  return is
end

function siteinds(tn::AbstractITensorNetwork)
  is = IndsNetwork(underlying_graph(tn))
  for v in vertices(tn)
    is[v] = uniqueinds(tn, v)
  end
  return is
end

function linkinds(tn::AbstractITensorNetwork)
  is = IndsNetwork(underlying_graph(tn))
  for e in edges(tn)
    is[e] = commoninds(tn, e)
  end
  return is
end

#
# Index access
#

function neighbor_itensors(tn::AbstractITensorNetwork, vertex...)
  return [tn[vn] for vn in neighbors(tn, vertex...)]
end

function uniqueinds(tn::AbstractITensorNetwork, vertex...)
  return uniqueinds(tn[vertex...], neighbor_itensors(tn, vertex...)...)
end

function uniqueinds(tn::AbstractITensorNetwork, edge::AbstractEdge)
  return uniqueinds(tn[src(edge)], tn[dst(edge)])
end

function uniqueinds(tn::AbstractITensorNetwork, edge::Pair)
  return uniqueinds(tn, edgetype(tn)(edge))
end

function siteinds(tn::AbstractITensorNetwork, vertex...)
  return uniqueinds(tn, vertex...)
end

function siteinds(::typeof(all), tn::AbstractITensorNetwork, vertex...)
  return siteinds(tn, vertex...)
end

function siteinds(::typeof(only), tn::AbstractITensorNetwork, vertex...)
  return only(siteinds(tn, vertex...))
end

function commoninds(tn::AbstractITensorNetwork, edge)
  e = NamedDimEdge(edge)
  return commoninds(tn[src(e)], tn[dst(e)])
end

function linkinds(tn::AbstractITensorNetwork, edge)
  return commoninds(tn, edge)
end

function linkinds(::typeof(all), tn::AbstractITensorNetwork, edge)
  return linkinds(tn, edge)
end

function linkinds(::typeof(only), tn::AbstractITensorNetwork, edge)
  return only(linkinds(tn, edge))
end

# Priming and tagging (changing Index identifiers)
function replaceinds(tn::AbstractITensorNetwork, is_is′::Pair{<:IndsNetwork,<:IndsNetwork})
  tn = copy(tn)
  is, is′ = is_is′
  @assert underlying_graph(is) == underlying_graph(is′)
  for v in vertices(is)
    isassigned(is, v) || continue
    setindex_preserve_graph!(tn, replaceinds(tn[v], is[v] => is′[v]), v)
  end
  for e in edges(is)
    isassigned(is, e) || continue
    for v in (src(e), dst(e))
      setindex_preserve_graph!(tn, replaceinds(tn[v], is[e] => is′[e]), v)
    end
  end
  return tn
end

function map_inds(f, tn::AbstractITensorNetwork, args...; kwargs...)
  is = IndsNetwork(tn)
  is′ = map_inds(f, is, args...; kwargs...)
  return replaceinds(tn, is => is′)
end

const map_inds_label_functions = [
  :prime,
  :setprime,
  :noprime,
  :replaceprime,
  :swapprime, # TODO: fix this one (broken)
  :addtags,
  :removetags,
  :replacetags,
  :settags,
  :sim,
  :swaptags,
  # :replaceind,
  # :replaceinds,
  # :swapind,
  # :swapinds,
]

for f in map_inds_label_functions
  @eval begin
    function $f(n::Union{IndsNetwork,AbstractITensorNetwork}, args...; kwargs...)
      return map_inds($f, n, args...; kwargs...)
    end

    function $f(
      ffilter::typeof(linkinds),
      n::Union{IndsNetwork,AbstractITensorNetwork},
      args...;
      kwargs...,
    )
      return map_inds($f, n, args...; sites=[], kwargs...)
    end

    function $f(
      ffilter::typeof(siteinds),
      n::Union{IndsNetwork,AbstractITensorNetwork},
      args...;
      kwargs...,
    )
      return map_inds($f, n, args...; links=[], kwargs...)
    end
  end
end

adjoint(tn::Union{IndsNetwork,AbstractITensorNetwork}) = prime(tn)

dag(tn::AbstractITensorNetwork) = map_vertex_data(dag, tn)

# TODO: should this make sure that internal indices
# don't clash?
function ⊗(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; kwargs...)
  return ⊔(tn1, tn2; kwargs...)
end

# TODO: remove this in favor of `inner_network` defined below?
# seems better if `inner` returns a number for every concrete AbstractITensorNetwork subtype
# 
# # TODO: name `inner_network` to denote it is lazy?
# # TODO: should this make sure that internal indices
# # don't clash?
# function inner(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
#   return dag(tn1) ⊗ tn2
# end

# TODO: how to define this lazily?
#norm(tn::AbstractITensorNetwork) = sqrt(inner(tn, tn))

function contract(tn::AbstractITensorNetwork; sequence=vertices(tn), kwargs...)
  sequence_linear_index = deepmap(v -> vertex_to_parent_vertex(tn)[v], sequence)
  return contract(Vector{ITensor}(tn); sequence=sequence_linear_index, kwargs...)
end

function contract!(tn::AbstractITensorNetwork, edge::Pair)
  return contract!(tn, edgetype(tn)(edge))
end

# Contract the tensors at vertices `src(edge)` and `dst(edge)`
# and store the results in the vertex `dst(edge)`, removing
# the vertex `src(edge)`.
# TODO: write this in terms of a more generic function
# `Graphs.merge_vertices!` (https://github.com/mtfishman/ITensorNetworks.jl/issues/12)
function contract!(tn::AbstractITensorNetwork, edge::AbstractEdge)
  neighbors_src = setdiff(neighbors(tn, src(edge)), [dst(edge)])
  neighbors_dst = setdiff(neighbors(tn, dst(edge)), [src(edge)])
  new_itensor = tn[src(edge)] * tn[dst(edge)]
  rem_vertex!(tn, src(edge))
  for n_src in neighbors_src
    add_edge!(tn, dst(edge) => n_src)
  end
  for n_dst in neighbors_dst
    add_edge!(tn, dst(edge) => n_dst)
  end
  # tn[dst(edge)] = new_itensor
  setindex_preserve_graph!(tn, new_itensor, dst(edge))
  return tn
end

function contract(tn::AbstractITensorNetwork, edge)
  return contract!(copy(tn), edge)
end

function tags(tn::AbstractITensorNetwork, edge)
  is = linkinds(tn, edge)
  return commontags(is)
end

function svd!(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
  return svd!(tn, edgetype(tn)(edge))
end

function svd!(
  tn::AbstractITensorNetwork,
  edge::AbstractEdge;
  U_vertex=src(edge),
  S_vertex=("S", edge),
  V_vertex=("V", edge),
  u_tags=tags(tn, edge),
  v_tags=tags(tn, edge),
  kwargs...,
)
  left_inds = uniqueinds(tn, edge)
  U, S, V = svd(tn[src(edge)], left_inds; lefttags=u_tags, right_tags=v_tags, kwargs...)

  rem_vertex!(tn, src(edge)) # TODO: avoid this if we can?
  add_vertex!(tn, U_vertex)
  tn[U_vertex] = U

  add_vertex!(tn, S_vertex)
  tn[S_vertex] = S

  add_vertex!(tn, V_vertex)
  tn[V_vertex] = V

  return tn
end

function svd(tn::AbstractITensorNetwork, edge; kwargs...)
  return svd!(copy(tn), edge; kwargs...)
end

function qr!(
  tn::AbstractITensorNetwork,
  edge::AbstractEdge;
  Q_vertex=src(edge),
  R_vertex=("R", edge),
  tags=tags(tn, edge),
  kwargs...,
)
  left_inds = uniqueinds(tn, edge)
  Q, R = factorize(tn[src(edge)], left_inds; tags, kwargs...)

  rem_vertex!(tn, src(edge)) # TODO: avoid this if we can?
  add_vertex!(tn, Q_vertex)
  tn[Q_vertex] = Q

  add_vertex!(tn, R_vertex)
  tn[R_vertex] = R

  return tn
end

function qr(tn::AbstractITensorNetwork, edge; kwargs...)
  return qr!(copy(tn), edge; kwargs...)
end

function factorize!(
  tn::AbstractITensorNetwork,
  edge::AbstractEdge;
  X_vertex=src(edge),
  Y_vertex=("Y", edge),
  tags=tags(tn, edge),
  kwargs...,
)
  neighbors_X = setdiff(neighbors(tn, src(edge)), [dst(edge)])
  left_inds = uniqueinds(tn, edge)
  X, Y = factorize(tn[src(edge)], left_inds; tags, kwargs...)

  rem_vertex!(tn, src(edge)) # TODO: avoid this if we can?
  add_vertex!(tn, X_vertex)
  add_vertex!(tn, Y_vertex)

  add_edge!(tn, X_vertex => Y_vertex)
  for nX in neighbors_X
    add_edge!(tn, X_vertex => nX)
  end
  add_edge!(tn, Y_vertex => dst(edge))

  # tn[X_vertex] = X
  setindex_preserve_graph!(tn, X, X_vertex)

  # tn[Y_vertex] = Y
  setindex_preserve_graph!(tn, Y, Y_vertex)

  return tn
end

function factorize(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
  return factorize!(copy(tn), edge; kwargs...)
end

# For ambiguity error; TODO: decide whether to use graph mutating methods when resulting graph is unchanged?
function _orthogonalize_edge!(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
  # factorize!(tn, edge; kwargs...)
  # new_vertex = only(neighbors(tn, src(edge)) ∩ neighbors(tn, dst(edge)))
  # contract!(tn, new_vertex => dst(edge))
  # return tn
  left_inds = uniqueinds(tn, edge)
  ltags = tags(tn, edge)
  X, Y = factorize(tn[src(edge)], left_inds; tags=ltags, ortho="left", kwargs...)
  tn[src(edge)] = X
  tn[dst(edge)] *= Y
  return tn
end

function orthogonalize!(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
  return _orthogonalize_edge!(tn, edge; kwargs...)
end

function orthogonalize!(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
  return orthogonalize!(tn, edgetype(tn)(edge); kwargs...)
end

function orthogonalize(tn::AbstractITensorNetwork, edge; kwargs...)
  return orthogonalize!(copy(tn), edge; kwargs...)
end

# TODO: decide whether to use graph mutating methods when resulting graph is unchanged?
function _truncate_edge!(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
  left_inds = uniqueinds(tn, edge)
  ltags = tags(tn, edge)
  U, S, V = svd(tn[src(edge)], left_inds; lefttags=ltags, ortho="left", kwargs...)
  tn[src(edge)] = U
  tn[dst(edge)] *= (S * V)
  return tn
end

function truncate!(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
  return _truncate_edge!(tn, edge; kwargs...)
end

function truncate!(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
  return truncate!(tn, edgetype(tn)(edge); kwargs...)
end

function truncate(tn::AbstractITensorNetwork, edge; kwargs...)
  return truncate!(copy(tn), edge; kwargs...)
end

# Orthogonalize an ITensorNetwork towards a source vertex, treating
# the network as a tree spanned by a spanning tree.
function orthogonalize(ψ::AbstractITensorNetwork, source_vertex::Tuple)
  spanning_tree_edges = post_order_dfs_edges(bfs_tree(ψ, source_vertex), source_vertex)
  for e in spanning_tree_edges
    ψ = orthogonalize(ψ, e)
  end
  return ψ
end

function Base.:*(c::Number, ψ::AbstractITensorNetwork)
  v₁ = first(vertices(ψ))
  cψ = copy(ψ)
  cψ[v₁] *= c
  return cψ
end

# TODO: should this make sure that internal indices
# don't clash?
function hvncat(
  dim::Int, tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; new_dim_names=(1, 2)
)
  dg = hvncat(dim, data_graph(tn1), data_graph(tn2); new_dim_names)

  # Add in missing edges that may be shared
  # across `tn1` and `tn2`.
  vertices1 = vertices(dg)[1:nv(tn1)]
  vertices2 = vertices(dg)[(nv(tn1) + 1):end]
  for v1 in vertices1, v2 in vertices2
    if hascommoninds(dg[v1], dg[v2])
      add_edge!(dg, v1 => v2)
    end
  end

  # TODO: Allow customization of the output type.
  ## return promote_type(typeof(tn1), typeof(tn2))(dg)
  ## return contract_output(typeof(tn1), typeof(tn2))(dg)

  return ITensorNetwork(dg)
end

# Return a list of vertices in the ITensorNetwork `ψ`
# that share indices with the ITensor `T`
function neighbor_vertices(ψ::AbstractITensorNetwork, T::ITensor)
  ψT = ψ ⊔ ITensorNetwork([T])
  v⃗ = neighbors(ψT, (2, 1))
  return Base.tail.(v⃗)
end

function inner_network(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; kwargs...)
  tn1 = sim(tn1; sites=[])
  tn2 = sim(tn2; sites=[])
  return ⊗(dag(tn1), tn2; kwargs...)
end

function norm_network(tn::AbstractITensorNetwork; kwargs...)
  return inner_network(tn, tn; kwargs...)
end

function flattened_inner_network(ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork)
  tn = inner(prime(ϕ; sites=[]), ψ)
  for v in vertices(ψ)
    tn = contract(tn, (2, v...) => (1, v...))
  end
  return tn
end

function contract_inner(
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  sequence=nothing,
  contraction_sequence_kwargs=(;),
)
  tn = inner(prime(ϕ; sites=[]), ψ)
  # TODO: convert to an IndsNetwork and compute the contraction sequence
  for v in vertices(ψ)
    tn = contract(tn, (2, v...) => (1, v...))
  end
  if isnothing(sequence)
    sequence = contraction_sequence(tn; contraction_sequence_kwargs...)
  end
  return contract(tn; sequence)[]
end

# TODO: rename `sqnorm` to match https://github.com/JuliaStats/Distances.jl?
norm2(ψ::AbstractITensorNetwork; sequence) = contract_inner(ψ, ψ; sequence)

#
# Printing
#

function show(io::IO, mime::MIME"text/plain", graph::AbstractITensorNetwork)
  println(io, "$(typeof(graph)) with $(nv(graph)) vertices:")
  show(io, mime, vertices(graph))
  println(io, "\n")
  println(io, "and $(ne(graph)) edge(s):")
  for e in edges(graph)
    show(io, mime, e)
    println(io)
  end
  println(io)
  println(io, "with vertex data:")
  show(io, mime, inds.(vertex_data(graph)))
  return nothing
end

show(io::IO, graph::AbstractITensorNetwork) = show(io, MIME"text/plain"(), graph)

function visualize(
  tn::AbstractITensorNetwork,
  args...;
  vertex_labels_prefix=nothing,
  vertex_labels=nothing,
  kwargs...,
)
  if !isnothing(vertex_labels_prefix)
    vertex_labels = [vertex_labels_prefix * string(v) for v in vertices(tn)]
  end
  return visualize(Vector{ITensor}(tn), args...; vertex_labels, kwargs...)
end

# 
# Link dimensions
# 

function maxlinkdim(tn::AbstractITensorNetwork)
  md = 1
  for e in edges(tn)
    md = max(md, linkdim(tn, e))
  end
  return md
end

function linkdim(tn::AbstractITensorNetwork, edge::Pair)
  return linkdim(tn, edgetype(tn)(edge))
end

function linkdim(tn::AbstractITensorNetwork, edge::AbstractEdge)
  ls = linkinds(tn, edge)
  return prod([isnothing(l) ? 1 : dim(l) for l in ls])
end

function linkdims(tn::AbstractITensorNetwork)
  return Dictionary(edges(tn), map(e -> linkdim(tn, e), edges(tn)))
end

# 
# Common index checking
# 

function hascommoninds(
  ::typeof(siteinds), A::AbstractITensorNetwork, B::AbstractITensorNetwork
)
  for v in vertices(A)
    !hascommoninds(siteinds(A, v), siteinds(B, v)) && return false
  end
  return true
end

function check_hascommoninds(
  ::typeof(siteinds), A::AbstractITensorNetwork, B::AbstractITensorNetwork
)
  N = nv(A)
  if nv(B) ≠ N
    throw(
      DimensionMismatch(
        "$(typeof(A)) and $(typeof(B)) have mismatched number of vertices $N and $(nv(B))."
      ),
    )
  end
  for v in vertices(A)
    !hascommoninds(siteinds(A, v), siteinds(B, v)) && error(
      "$(typeof(A)) A and $(typeof(B)) B must share site indices. On vertex $v, A has site indices $(siteinds(A, v)) while B has site indices $(siteinds(B, v)).",
    )
  end
  return nothing
end

function hassameinds(
  ::typeof(siteinds), A::AbstractITensorNetwork, B::AbstractITensorNetwork
)
  nv(A) ≠ nv(B) && return false
  for v in vertices(A)
    !ITensors.hassameinds(siteinds(all, A, v), siteinds(all, B, v)) && return false
  end
  return true
end

# 
# Site combiners
# 

function site_combiners(tn::AbstractITensorNetwork)
  Cs = NamedDimDataGraph{ITensor}(copy(underlying_graph(tn)))
  for v in vertices(tn)
    s = siteinds(all, tn, v)
    Cs[v] = combiner(s; tags=commontags(s))
  end
  return Cs
end
