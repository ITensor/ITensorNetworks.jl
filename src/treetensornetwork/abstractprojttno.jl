abstract type AbstractProjTTNO end

copy(::AbstractProjTTNO) = error("Not implemented")

set_nsite!(::AbstractProjTTNO, nsite) = error("Not implemented")

make_environment!(::AbstractProjTTNO, psi, e) = error("Not implemented")

underlying_graph(P::AbstractProjTTNO) = underlying_graph(P.H)

pos(P::AbstractProjTTNO) = P.pos

Graphs.edgetype(P::AbstractProjTTNO) = edgetype(underlying_graph(P))

on_edge(P::AbstractProjTTNO) = isa(pos(P), edgetype(P))

nsite(P::AbstractProjTTNO) = on_edge(P) ? 0 : length(pos(P))

function sites(P::AbstractProjTTNO)
  on_edge(P) && return eltype(underlying_graph(P))[]
  return pos(P)
end

function incident_edges(P::AbstractProjTTNO)::Vector{NamedDimEdge{Tuple}}
  on_edge(P) && return [pos(P), reverse(pos(P))]
  edges = [
    [edgetype(P)(n => v) for n in setdiff(neighbors(underlying_graph(P), v), sites(P))] for
    v in sites(P)
  ]
  return collect(Base.Iterators.flatten(edges))
end

function internal_edges(P::AbstractProjTTNO)::Vector{NamedDimEdge{Tuple}}
  on_edge(P) && return edgetype(P)[]
  edges = [
    [edgetype(P)(v => n) for n in neighbors(underlying_graph(P), v) ∩ sites(P)] for
    v in sites(P)
  ]
  return collect(Base.Iterators.flatten(edges))
end

function environment(P::AbstractProjTTNO, edge::NamedDimEdge{Tuple})::ITensor
  return P.environments[edge]
end

# there has to be a better way to do this...
function _separate_first(V::Vector)
  sep = Base.Iterators.peel(V)
  isnothing(sep) && return eltype(V)[], eltype(V)[]
  return sep[1], collect(sep[2])
end

function _separate_first_two(V::Vector)
  frst, rst = _separate_first(V)
  scnd, rst = _separate_first(rst)
  return frst, scnd, rst
end

function contract(P::AbstractProjTTNO, v::ITensor)::ITensor
  environments = ITensor[environment(P, edge) for edge in incident_edges(P)]
  # manual heuristic for contraction order fixing: for each site in ProjTTNO, apply up to
  # two environments, then TTNO tensor, then other environments
  if on_edge(P)
    itensor_map = environments
  else
    itensor_map = Union{ITensor,OneITensor}[] # TODO: will a Hamiltonian TTNO tensor ever be a OneITensor?
    for s in sites(P)
      site_envs = filter(hascommoninds(P.H[s]), environments)
      frst, scnd, rst = _separate_first_two(site_envs)
      site_tensors = vcat(frst, scnd, P.H[s], rst)
      append!(itensor_map, site_tensors)
    end
  end
  # TODO: actually use optimal contraction sequence here
  Hv = v
  for it in itensor_map
    Hv *= it
  end
  return Hv
end

function product(P::AbstractProjTTNO, v::ITensor)::ITensor
  Pv = contract(P, v)
  if order(Pv) != order(v)
    error(
      string(
        "The order of the ProjTTNO-ITensor product P*v is not equal to the order of the ITensor v, ",
        "this is probably due to an index mismatch.\nCommon reasons for this error: \n",
        "(1) You are trying to multiply the ProjTTNO with the $(nsite(P))-site wave-function at the wrong position.\n",
        "(2) `orthogonalize!` was called, changing the MPS without updating the ProjTTNO.\n\n",
        "P*v inds: $(inds(Pv)) \n\n",
        "v inds: $(inds(v))",
      ),
    )
  end
  return noprime(Pv)
end

(P::AbstractProjTTNO)(v::ITensor) = product(P, v)

function Base.eltype(P::AbstractProjTTNO)::Type
  ElType = eltype(P.H(first(sites(P))))
  for v in sites(P)
    ElType = promote_type(ElType, eltype(P.H[v]))
  end
  for e in incident_edges(P)
    ElType = promote_type(ElType, eltype(environments(P, e)))
  end
  return ElType
end

function Base.size(P::AbstractProjTTNO)::Tuple{Int,Int}
  d = 1
  for e in incident_edges(P)
    for i in inds(environment(P, e))
      plev(i) > 0 && (d *= dim(i))
    end
  end
  for j in sites(P)
    for i in inds(P.H[j])
      plev(i) > 0 && (d *= dim(i))
    end
  end
  return (d, d)
end

function position!(
  P::AbstractProjTTNO, psi::TTNS, pos::Union{Vector{<:Tuple},NamedDimEdge{Tuple}}
)
  # shift position
  P.pos = pos
  # invalidate environments corresponding to internal edges
  for e in internal_edges(P)
    unset!(P.environments, e)
  end
  # make all environments surrounding new position
  for e in incident_edges(P)
    make_environment!(P, psi, e)
  end
  return P
end
