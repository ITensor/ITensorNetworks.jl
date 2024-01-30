abstract type AbstractProjTTN{V} end

environments(::AbstractProjTTN) = error("Not implemented")
operator(::AbstractProjTTN) = error("Not implemented")
pos(::AbstractProjTTN) = error("Not implemented")

underlying_graph(P::AbstractProjTTN) = error("Not implemented")

copy(::AbstractProjTTN) = error("Not implemented")

set_nsite(::AbstractProjTTN, nsite) = error("Not implemented")

# silly constructor wrapper
shift_position(::AbstractProjTTN, pos) = error("Not implemented")

set_environments(p::AbstractProjTTN, environments) = error("Not implemented")
set_environment(p::AbstractProjTTN, edge, environment) = error("Not implemented")
make_environment!(P::AbstractProjTTN, psi, e) = error("Not implemented")
make_environment(P::AbstractProjTTN, psi, e) = error("Not implemented")

Graphs.edgetype(P::AbstractProjTTN) = edgetype(underlying_graph(P))

on_edge(P::AbstractProjTTN) = isa(pos(P), edgetype(P))

nsite(P::AbstractProjTTN) = on_edge(P) ? 0 : length(pos(P))

function sites(P::AbstractProjTTN{V}) where {V}
  on_edge(P) && return V[]
  return pos(P)
end

function incident_edges(P::AbstractProjTTN{V})::Vector{NamedEdge{V}} where {V}
  on_edge(P) && return [pos(P), reverse(pos(P))]
  edges = [
    [edgetype(P)(n => v) for n in setdiff(neighbors(underlying_graph(P), v), sites(P))] for
    v in sites(P)
  ]
  return collect(Base.Iterators.flatten(edges))
end

function internal_edges(P::AbstractProjTTN{V})::Vector{NamedEdge{V}} where {V}
  on_edge(P) && return edgetype(P)[]
  edges = [
    [edgetype(P)(v => n) for n in neighbors(underlying_graph(P), v) ∩ sites(P)] for
    v in sites(P)
  ]
  return collect(Base.Iterators.flatten(edges))
end

environment(P::AbstractProjTTN, edge::Pair) = environment(P, edgetype(P)(edge))
function environment(P::AbstractProjTTN, edge::AbstractEdge)
  return environments(P)[edge]
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

function contract(P::AbstractProjTTN, v::ITensor)::ITensor
  environments = ITensor[environment(P, edge) for edge in incident_edges(P)]
  # manual heuristic for contraction order fixing: for each site in ProjTTN, apply up to
  # two environments, then TTN tensor, then other environments
  if on_edge(P)
    itensor_map = environments
  else
    itensor_map = Union{ITensor,OneITensor}[] # TODO: will a Hamiltonian TTN tensor ever be a OneITensor?
    for s in sites(P)
      site_envs = filter(hascommoninds(operator(P)[s]), environments)
      frst, scnd, rst = _separate_first_two(site_envs)
      site_tensors = vcat(frst, scnd, operator(P)[s], rst)
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

function product(P::AbstractProjTTN, v::ITensor)::ITensor
  Pv = contract(P, v)
  if order(Pv) != order(v)
    error(
      string(
        "The order of the ProjTTN-ITensor product P*v is not equal to the order of the ITensor v, ",
        "this is probably due to an index mismatch.\nCommon reasons for this error: \n",
        "(1) You are trying to multiply the ProjTTN with the $(nsite(P))-site wave-function at the wrong position.\n",
        "(2) `orthogonalize!` was called, changing the MPS without updating the ProjTTN.\n\n",
        "P*v inds: $(inds(Pv)) \n\n",
        "v inds: $(inds(v))",
      ),
    )
  end
  return noprime(Pv)
end

(P::AbstractProjTTN)(v::ITensor) = product(P, v)

function Base.eltype(P::AbstractProjTTN)::Type
  ElType = eltype(operator(P)(first(sites(P))))
  for v in sites(P)
    ElType = promote_type(ElType, eltype(operator(P)[v]))
  end
  for e in incident_edges(P)
    ElType = promote_type(ElType, eltype(environments(P, e)))
  end
  return ElType
end

function Base.size(P::AbstractProjTTN)::Tuple{Int,Int}
  d = 1
  for e in incident_edges(P)
    for i in inds(environment(P, e))
      plev(i) > 0 && (d *= dim(i))
    end
  end
  for j in sites(P)
    for i in inds(operator(P)[j])
      plev(i) > 0 && (d *= dim(i))
    end
  end
  return (d, d)
end

function position(P::AbstractProjTTN, psi::AbstractTTN, pos)
  P = shift_position(P, pos)
  P = invalidate_environments(P)
  P = make_environments(P, psi)
  return P
end

function invalidate_environments(P::AbstractProjTTN)
  ie = internal_edges(P)
  newenvskeys = filter(!in(ie), keys(environments(P)))
  P = set_environments(P, getindices(environments(P), newenvskeys))
  return P
end

function invalidate_environment(P::AbstractProjTTN, e::AbstractEdge)
  newenvskeys = filter(!isequal(e), keys(environments(P)))
  P = set_environments(P, getindices(environments(P), newenvskeys))
  return P
end

function make_environments(P::AbstractProjTTN, psi::AbstractTTN)
  for e in incident_edges(P)
    P = make_environment(P, psi, e)
  end
  return P
end
