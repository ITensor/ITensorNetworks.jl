# TODO: Replace `AbstractITensorNetwork` with a trait `IsTree`.
abstract type AbstractTreeTensorNetwork{V} <: AbstractITensorNetwork{V} end

function underlying_graph_type(G::Type{<:AbstractTreeTensorNetwork})
  return underlying_graph_type(data_graph_type(G))
end

# 
# Field access
# 

ITensorNetwork(ψ::AbstractTreeTensorNetwork) = ψ.itensor_network
ortho_center(ψ::AbstractTreeTensorNetwork) = ψ.ortho_center

function default_root_vertex(gs::AbstractGraph...)
  # @assert all(is_tree.(gs))
  return first(leaf_vertices(gs[end]))
end

# 
# Orthogonality center
# 

isortho(ψ::AbstractTreeTensorNetwork) = isone(length(ortho_center(ψ)))

function set_ortho_center(
  ψ::AbstractTreeTensorNetwork{V}, new_center::Vector{<:V}
) where {V}
  return typeof(ψ)(itensor_network(ψ), new_center)
end

reset_ortho_center(ψ::AbstractTreeTensorNetwork) = set_ortho_center(ψ, vertices(ψ))

# 
# Dense constructors
# 

# construct from dense ITensor, using IndsNetwork of site indices
function (::Type{TTNT})(
  A::ITensor, is::IndsNetwork; ortho_center=default_root_vertex(is), kwargs...
) where {TTNT<:AbstractTreeTensorNetwork}
  for v in vertices(is)
    @assert hasinds(A, is[v])
  end
  @assert ortho_center ∈ vertices(is)
  ψ = ITensorNetwork(is)
  Ã = A
  for e in post_order_dfs_edges(ψ, ortho_center)
    left_inds = uniqueinds(is, e)
    L, R = factorize(Ã, left_inds; tags=edge_tag(e), ortho="left", kwargs...)
    l = commonind(L, R)
    ψ[src(e)] = L
    is[e] = [l]
    Ã = R
  end
  ψ[ortho_center] = Ã
  T = TTNT(ψ)
  T = orthogonalize(T, ortho_center)
  return T
end

# construct from dense ITensor, using AbstractNamedGraph and vector of site indices
# TODO: remove if it doesn't turn out to be useful
function (::Type{TTNT})(
  A::ITensor, sites::Vector, g::AbstractNamedGraph; vertex_order=vertices(g), kwargs...
) where {TTNT<:AbstractTreeTensorNetwork}
  is = IndsNetwork(g; site_space=Dictionary(vertex_order, sites))
  return TTNT(A, is; kwargs...)
end

# construct from dense array, using IndsNetwork
# TODO: probably remove this one, doesn't seem very useful
function (::Type{TTNT})(
  A::AbstractArray{<:Number}, is::IndsNetwork; vertex_order=vertices(is), kwargs...
) where {TTNT<:AbstractTreeTensorNetwork}
  sites = [is[v] for v in vertex_order]
  return TTNT(itensor(A, sites...), is; kwargs...)
end

# construct from dense array, using NamedDimGraph and vector of site indices
function (::Type{TTNT})(
  A::AbstractArray{<:Number}, sites::Vector, args...; kwargs...
) where {TTNT<:AbstractTreeTensorNetwork}
  return TTNT(itensor(A, sites...), sites, args...; kwargs...)
end

# 
# Orthogonalization
# 

function orthogonalize(ψ::AbstractTreeTensorNetwork{V}, root_vertex::V; kwargs...) where {V}
  (isortho(ψ) && only(ortho_center(ψ)) == root_vertex) && return ψ
  if isortho(ψ)
    edge_list = edge_path(ψ, only(ortho_center(ψ)), root_vertex)
  else
    edge_list = post_order_dfs_edges(ψ, root_vertex)
  end
  for e in edge_list
    ψ = orthogonalize(ψ, e)
  end
  return set_ortho_center(ψ, [root_vertex])
end

# For ambiguity error

function orthogonalize(tn::AbstractTreeTensorNetwork, edge::AbstractEdge; kwargs...)
  return typeof(tn)(orthogonalize(ITensorNetwork(tn), edge; kwargs...))
end

# 
# Truncation
# 

function truncate(
  ψ::AbstractTreeTensorNetwork; root_vertex::Tuple=default_root_vertex(ψ), kwargs...
)
  for e in post_order_dfs_edges(ψ, root_vertex)
    # always orthogonalize towards source first to make truncations controlled
    ψ = orthogonalize(ψ, src(e))
    ψ = truncate(ψ, e; kwargs...)
    ψ = set_ortho_center(ψ, [dst(e)])
  end
  return ψ
end

# For ambiguity error
function truncate(ψ::AbstractTreeTensorNetwork, edge::AbstractEdge; kwargs...)
  return typeof(tn)(truncate(ITensorNetwork(tn), edge; kwargs...))
end

#
# Contraction
#

# TODO: decide on contraction order: reverse dfs vertices or forward dfs edges?
function contract(
  ψ::AbstractTreeTensorNetwork{V}, root_vertex::V=default_root_vertex(ψ); kwargs...
) where {V}
  ψ = copy(ψ)
  # reverse post order vertices
  traversal_order = reverse(post_order_dfs_vertices(ψ, root_vertex))
  return contract(ITensorNetwork(ψ); sequence=traversal_order, kwargs...)
  # # forward post order edges
  # ψ = copy(ψ)
  # for e in post_order_dfs_edges(ψ, root_vertex)
  #   ψ = contract(ψ, e)
  # end
  # return ψ[root_vertex]
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
  if isortho(ψ)
    return norm(ψ[only(ortho_center(ψ))])
  end
  return √(abs(real(inner(ψ, ψ))))
end

# 
# Utility
# 

function normalize!(ψ::AbstractTreeTensorNetwork)
  c = ortho_center(ψ)
  lognorm_ψ = lognorm(ψ)
  if lognorm_ψ == -Inf
    return ψ
  end
  z = exp(lognorm_ψ / length(c))
  for v in c
    ψ[v] ./= z
  end
  return ψ
end

function normalize(ψ::AbstractTreeTensorNetwork)
  return normalize!(copy(ψ))
end

function _apply_to_orthocenter!(f, ψ::AbstractTreeTensorNetwork, x)
  v = first(ortho_center(ψ))
  ψ[v] = f(ψ[v], x)
  return ψ
end

function _apply_to_orthocenter(f, ψ::AbstractTreeTensorNetwork, x)
  return _apply_to_orthocenter!(f, copy(ψ), x)
end

Base.:*(ψ::AbstractTreeTensorNetwork, α::Number) = _apply_to_orthocenter(*, ψ, α)

Base.:*(α::Number, ψ::AbstractTreeTensorNetwork) = ψ * α

Base.:/(ψ::AbstractTreeTensorNetwork, α::Number) = _apply_to_orthocenter(/, ψ, α)

Base.:-(ψ::AbstractTreeTensorNetwork) = -1 * ψ

function LinearAlgebra.rmul!(ψ::AbstractTreeTensorNetwork, α::Number)
  return _apply_to_orthocenter!(*, ψ, α)
end

function lognorm(ψ::AbstractTreeTensorNetwork)
  if isortho(ψ)
    return log(norm(ψ[only(ortho_center(ψ))]))
  end
  lognorm2_ψ = loginner(ψ, ψ)
  rtol = eps(real(scalartype(ψ))) * 10
  atol = rtol
  if !IsApprox.isreal(lognorm2_ψ, Approx(; rtol=rtol, atol=atol))
    @warn "log(norm²) is $lognorm2_T, which is not real up to a relative tolerance of $rtol and an absolute tolerance of $atol. Taking the real part, which may not be accurate."
  end
  return 0.5 * real(lognorm2_ψ)
end

function logdot(ψ1::TTNT, ψ2::TTNT; kwargs...) where {TTNT<:AbstractTreeTensorNetwork}
  return loginner(ψ1, ψ2; kwargs...)
end

# TODO: stick with this traversal or find optimal contraction sequence?
function loginner(
  ψ1::TTNT, ψ2::TTNT; root_vertex=default_root_vertex(ψ1, ψ2)
)::Number where {TTNT<:AbstractTreeTensorNetwork}
  N = nv(ψ1)
  if nv(ψ2) != N
    throw(DimensionMismatch("inner: mismatched number of vertices $N and $(nv(ψ2))"))
  end
  ψ1dag = sim(dag(ψ1); sites=[])
  traversal_order = reverse(post_order_dfs_vertices(ψ1, root_vertex))
  check_hascommoninds(siteinds, ψ1dag, ψ2)

  O = ψ1dag[root_vertex] * ψ2[root_vertex]

  normO = norm(O)
  log_inner_tot = log(normO)
  O ./= normO

  for v in traversal_order[2:end]
    O = (O * ψ1dag[v]) * ψ2[v]
    normO = norm(O)
    log_inner_tot += log(normO)
    O ./= normO
  end

  if !isreal(O[]) || real(O[]) < 0
    log_inner_tot += log(complex(O[]))
  end
  return log_inner_tot
end

function _add_maxlinkdims(ψs::AbstractTreeTensorNetwork...)
  maxdims = Dictionary{edgetype(ψs[1]),Int}()
  for e in edges(ψs[1])
    maxdims[e] = sum(ψ -> linkdim(ψ, e), ψs)
    maxdims[reverse(e)] = maxdims[e]
  end
  return maxdims
end

# TODO: actually implement this?
function Base.:+(
  ::ITensors.Algorithm"densitymatrix",
  ψs::TTNT...;
  cutoff=1e-15,
  root_vertex=default_root_vertex(ψs...),
  kwargs...,
) where {TTNT<:AbstractTreeTensorNetwork}
  return error("Not implemented (yet) for trees.")
end

function Base.:+(
  ::ITensors.Algorithm"directsum", ψs::TTNT...; root_vertex=default_root_vertex(ψs...)
) where {TTNT<:AbstractTreeTensorNetwork}
  @assert all(ψ -> nv(first(ψs)) == nv(ψ), ψs)

  # Output state
  ϕ = TTNS(siteinds(ψs[1]))

  vs = post_order_dfs_vertices(ϕ, root_vertex)
  es = post_order_dfs_edges(ϕ, root_vertex)
  link_space = Dict{edgetype(ϕ),Index}()

  for v in reverse(vs)
    edges = filter(e -> dst(e) == v || src(e) == v, es)
    dims_in = findall(e -> dst(e) == v, edges)
    dim_out = findfirst(e -> src(e) == v, edges)

    ls = [Tuple(only(linkinds(ψ, e)) for e in edges) for ψ in ψs]
    ϕv, lv = directsum((ψs[i][v] => ls[i] for i in 1:length(ψs))...; tags=tags.(first(ls)))
    for din in dims_in
      link_space[edges[din]] = lv[din]
    end
    if !isnothing(dim_out)
      ϕv = replaceind(ϕv, lv[dim_out] => dag(link_space[edges[dim_out]]))
    end

    ϕ[v] = ϕv
  end
  return convert(TTNT, ϕ)
end

# TODO: switch default algorithm once more are implemented
function Base.:+(
  ψs::AbstractTreeTensorNetwork...; alg=ITensors.Algorithm"directsum"(), kwargs...
)
  return +(ITensors.Algorithm(alg), ψs...; kwargs...)
end

Base.:+(ψ::AbstractTreeTensorNetwork) = ψ

ITensors.add(ψs::AbstractTreeTensorNetwork...; kwargs...) = +(ψs...; kwargs...)

function Base.:-(ψ1::AbstractTreeTensorNetwork, ψ2::AbstractTreeTensorNetwork; kwargs...)
  return +(ψ1, -ψ2; kwargs...)
end

function ITensors.add(A::T, B::T; kwargs...) where {T<:AbstractTreeTensorNetwork}
  return +(A, B; kwargs...)
end

function permute(
  ψ::TTNT, ::Tuple{typeof(linkind),typeof(siteinds),typeof(linkind)}
)::TTNT where {TTNT<:AbstractTreeTensorNetwork}
  ψ̃ = TTNT(underlying_graph(ψ))
  for v in vertices(ψ)
    ls = [only(linkinds(ψ, n => v)) for n in neighbors(ψ, v)] # TODO: won't work for multiple indices per link...
    ss = sort(Tuple(siteinds(ψ, v)); by=plev)
    setindex_preserve_graph!(
      ψ̃, permute(ψ[v], filter(!isnothing, (ls[1], ss..., ls[2:end]...))), v
    )
  end
  ψ̃ = set_ortho_center(ψ̃, ortho_center(ψ))
  return ψ̃
end

function Base.isapprox(
  x::AbstractTreeTensorNetwork,
  y::AbstractTreeTensorNetwork;
  atol::Real=0,
  rtol::Real=Base.rtoldefault(
    LinearAlgebra.promote_leaf_eltypes(x), LinearAlgebra.promote_leaf_eltypes(y), atol
  ),
)
  d = norm(x - y)
  if isfinite(d)
    return d <= max(atol, rtol * max(norm(x), norm(y)))
  else
    error("In `isapprox(x::TTNS, y::TTNS)`, `norm(x - y)` is not finite")
  end
end
