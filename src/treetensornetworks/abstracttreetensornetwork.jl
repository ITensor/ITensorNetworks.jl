using Graphs: has_vertex
using NamedGraphs.GraphsExtensions:
  GraphsExtensions,
  edge_path,
  leaf_vertices,
  post_order_dfs_edges,
  post_order_dfs_vertices,
  a_star
using NamedGraphs: namedgraph_a_star
using IsApprox: IsApprox, Approx
using ITensors: @Algorithm_str, directsum, hasinds, permute, plev
using ITensorMPS: linkind, loginner, lognorm, orthogonalize

abstract type AbstractTreeTensorNetwork{V} <: AbstractITensorNetwork{V} end

const AbstractTTN = AbstractTreeTensorNetwork

function DataGraphs.underlying_graph_type(G::Type{<:AbstractTTN})
  return underlying_graph_type(data_graph_type(G))
end

# 
# Field access
# 

ITensorNetwork(tn::AbstractTTN) = error("Not implemented")
ortho_region(tn::AbstractTTN) = error("Not implemented")

# 
# Orthogonality center
# 

function set_ortho_region(tn::AbstractTTN, new_region)
  return error("Not implemented")
end

function ITensorMPS.orthogonalize(ttn::AbstractTTN, region::Vector; kwargs...)
  new_path = post_order_dfs_edges_region(ttn, region)
  existing_path = post_order_dfs_edges_region(ttn, ortho_region(ttn))
  path = setdiff(new_path, existing_path)
  if !isempty(path)
    ttn = typeof(ttn)(orthogonalize(ITensorNetwork(ttn), path; kwargs...))
  end
  return set_ortho_region(ttn, region)
end

# 
# Truncation
# 

function Base.truncate(
  tn::AbstractTTN; root_vertex=GraphsExtensions.default_root_vertex(tn), kwargs...
)
  for e in post_order_dfs_edges(tn, root_vertex)
    # always orthogonalize towards source first to make truncations controlled
    tn = orthogonalize(tn, src(e))
    tn = truncate(tn, e; kwargs...)
    tn = set_ortho_region(tn, typeof(ortho_region(tn))([dst(e)]))
  end
  return tn
end

# For ambiguity error
function Base.truncate(tn::AbstractTTN, edge::AbstractEdge; kwargs...)
  return typeof(tn)(truncate(ITensorNetwork(tn), edge; kwargs...))
end

#
# Contraction
#

# TODO: decide on contraction order: reverse dfs vertices or forward dfs edges?
function NDTensors.contract(
  tn::AbstractTTN, root_vertex=GraphsExtensions.default_root_vertex(tn); kwargs...
)
  tn = copy(tn)
  # reverse post order vertices
  traversal_order = reverse(post_order_dfs_vertices(tn, root_vertex))
  return contract(ITensorNetwork(tn); sequence=traversal_order, kwargs...)
  # # forward post order edges
  # tn = copy(tn)
  # for e in post_order_dfs_edges(tn, root_vertex)
  #   tn = contract(tn, e)
  # end
  # return tn[root_vertex]
end

function ITensors.inner(
  x::AbstractTTN, y::AbstractTTN; root_vertex=GraphsExtensions.default_root_vertex(x)
)
  xᴴ = sim(dag(x); sites=[])
  y = sim(y; sites=[])
  xy = xᴴ ⊗ y
  # TODO: find the largest tensor and use it as
  # the `root_vertex`.
  for e in post_order_dfs_edges(y, root_vertex)
    if has_vertex(xy, (src(e), 2))
      xy = contract(xy, (src(e), 2) => (src(e), 1))
    end
    xy = contract(xy, (src(e), 1) => (dst(e), 1))
    if has_vertex(xy, (dst(e), 2))
      xy = contract(xy, (dst(e), 2) => (dst(e), 1))
    end
  end
  return xy[root_vertex, 1][]
end

function LinearAlgebra.norm(tn::AbstractTTN)
  if isone(length(ortho_region(tn)))
    return norm(tn[only(ortho_region(tn))])
  end
  return √(abs(real(inner(tn, tn))))
end

# 
# Utility
# 

function LinearAlgebra.normalize!(tn::AbstractTTN)
  c = ortho_region(tn)
  lognorm_tn = lognorm(tn)
  if lognorm_tn == -Inf
    return tn
  end
  z = exp(lognorm_tn / length(c))
  for v in c
    tn[v] ./= z
  end
  return tn
end

function LinearAlgebra.normalize(tn::AbstractTTN)
  return normalize!(copy(tn))
end

function _apply_to_ortho_region!(f, tn::AbstractTTN, x)
  v = first(ortho_region(tn))
  tn[v] = f(tn[v], x)
  return tn
end

function _apply_to_ortho_region(f, tn::AbstractTTN, x)
  return _apply_to_ortho_region!(f, copy(tn), x)
end

Base.:*(tn::AbstractTTN, α::Number) = _apply_to_ortho_region(*, tn, α)

Base.:*(α::Number, tn::AbstractTTN) = tn * α

Base.:/(tn::AbstractTTN, α::Number) = _apply_to_ortho_region(/, tn, α)

Base.:-(tn::AbstractTTN) = -1 * tn

function LinearAlgebra.rmul!(tn::AbstractTTN, α::Number)
  return _apply_to_ortho_region!(*, tn, α)
end

function ITensorMPS.lognorm(tn::AbstractTTN)
  if isone(length(ortho_region(tn)))
    return log(norm(tn[only(ortho_region(tn))]))
  end
  lognorm2_tn = loginner(tn, tn)
  rtol = eps(real(scalartype(tn))) * 10
  atol = rtol
  if !IsApprox.isreal(lognorm2_tn, Approx(; rtol=rtol, atol=atol))
    @warn "log(norm²) is $lognorm2_T, which is not real up to a relative tolerance of $rtol and an absolute tolerance of $atol. Taking the real part, which may not be accurate."
  end
  return 0.5 * real(lognorm2_tn)
end

function logdot(tn1::AbstractTTN, tn2::AbstractTTN; kwargs...)
  return loginner(tn1, tn2; kwargs...)
end

# TODO: stick with this traversal or find optimal contraction sequence?
function ITensorMPS.loginner(
  tn1::AbstractTTN, tn2::AbstractTTN; root_vertex=GraphsExtensions.default_root_vertex(tn1)
)
  N = nv(tn1)
  if nv(tn2) != N
    throw(DimensionMismatch("inner: mismatched number of vertices $N and $(nv(tn2))"))
  end
  tn1dag = sim(dag(tn1); sites=[])
  traversal_order = reverse(post_order_dfs_vertices(tn1, root_vertex))

  O = tn1dag[root_vertex] * tn2[root_vertex]

  normO = norm(O)
  log_inner_tot = log(normO)
  O ./= normO

  for v in traversal_order[2:end]
    O = (O * tn1dag[v]) * tn2[v]
    normO = norm(O)
    log_inner_tot += log(normO)
    O ./= normO
  end

  if !isreal(O[]) || real(O[]) < 0
    log_inner_tot += log(complex(O[]))
  end
  return log_inner_tot
end

function _add_maxlinkdims(tns::AbstractTTN...)
  maxdims = Dictionary{edgetype(tns[1]),Int}()
  for e in edges(tns[1])
    maxdims[e] = sum(tn -> linkdim(tn, e), tns)
    maxdims[reverse(e)] = maxdims[e]
  end
  return maxdims
end

# TODO: actually implement this?
function Base.:+(
  ::Algorithm"densitymatrix",
  tns::AbstractTTN...;
  cutoff=1e-15,
  root_vertex=GraphsExtensions.default_root_vertex(first(tns)),
  kwargs...,
)
  return error("Not implemented (yet) for trees.")
end

function Base.:+(
  ::Algorithm"directsum",
  tns::AbstractTTN...;
  root_vertex=GraphsExtensions.default_root_vertex(first(tns)),
)
  @assert all(tn -> nv(first(tns)) == nv(tn), tns)

  # Output state
  tn = ttn(siteinds(tns[1]))

  vs = post_order_dfs_vertices(tn, root_vertex)
  es = post_order_dfs_edges(tn, root_vertex)
  link_space = Dict{edgetype(tn),Index}()

  for v in reverse(vs)
    edges = filter(e -> dst(e) == v || src(e) == v, es)
    dims_in = findall(e -> dst(e) == v, edges)
    dim_out = findfirst(e -> src(e) == v, edges)
    ls = [Tuple(only(linkinds(tn, e)) for e in edges) for tn in tns]
    tnv, lv = directsum(
      (tns[i][v] => ls[i] for i in 1:length(tns))...; tags=tags.(first(ls))
    )
    for din in dims_in
      link_space[edges[din]] = lv[din]
    end
    if !isnothing(dim_out)
      tnv = replaceind(tnv, lv[dim_out] => dag(link_space[edges[dim_out]]))
    end
    tn[v] = tnv
  end
  return tn
end

# TODO: switch default algorithm once more are implemented
function Base.:+(tns::AbstractTTN...; alg=Algorithm"directsum"(), kwargs...)
  return +(Algorithm(alg), tns...; kwargs...)
end

Base.:+(tn::AbstractTTN) = tn

ITensors.add(tns::AbstractTTN...; kwargs...) = +(tns...; kwargs...)

function Base.:-(tn1::AbstractTTN, tn2::AbstractTTN; kwargs...)
  return +(tn1, -tn2; kwargs...)
end

function ITensors.add(tn1::AbstractTTN, tn2::AbstractTTN; kwargs...)
  return +(tn1, tn2; kwargs...)
end

function Base.isapprox(
  x::AbstractTTN,
  y::AbstractTTN;
  atol::Real=0,
  rtol::Real=Base.rtoldefault(scalartype(x), scalartype(y), atol),
)
  d = norm(x - y)
  if isfinite(d)
    return d <= max(atol, rtol * max(norm(x), norm(y)))
  else
    error("In `isapprox(x::AbstractTTN, y::AbstractTTN)`, `norm(x - y)` is not finite")
  end
end

#
# Inner products
#

# TODO: implement using multi-graph disjoint union
function ITensors.inner(
  y::AbstractTTN,
  A::AbstractTTN,
  x::AbstractTTN;
  root_vertex=GraphsExtensions.default_root_vertex(x),
)
  traversal_order = reverse(post_order_dfs_vertices(x, root_vertex))
  ydag = sim(dag(y); sites=[])
  x = sim(x; sites=[])
  O = ydag[root_vertex] * A[root_vertex] * x[root_vertex]
  for v in traversal_order[2:end]
    O = O * ydag[v] * A[v] * x[v]
  end
  return O[]
end

# TODO: implement using multi-graph disjoint
function ITensors.inner(
  B::AbstractTTN,
  y::AbstractTTN,
  A::AbstractTTN,
  x::AbstractTTN;
  root_vertex=GraphsExtensions.default_root_vertex(B),
)
  N = nv(B)
  if nv(y) != N || nv(x) != N || nv(A) != N
    throw(
      DimensionMismatch(
        "inner: mismatched number of vertices $N and $(nv(x)) or $(nv(y)) or $(nv(A))"
      ),
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

function ITensorMPS.expect(
  operator::String,
  state::AbstractTTN;
  vertices=vertices(state),
  # TODO: verify that this is a sane default
  root_vertex=GraphsExtensions.default_root_vertex(state),
)
  # TODO: Optimize this with proper caching.
  state /= norm(state)
  sites = siteinds(state)
  ordered_vertices = reverse(post_order_dfs_vertices(sites, root_vertex))
  res = Dictionary(vertices, undef)
  for v in ordered_vertices
    !(v in vertices) && continue
    state = orthogonalize(state, v)
    @assert isone(length(sites[v]))
    #ToDo: Add compatibility with more than a single index per vertex
    op_v = op(operator, only(sites[v]))
    res[v] = (dag(state[v]) * apply(op_v, state[v]))[]
  end
  return mapreduce(typeof, promote_type, res).(res)
end
