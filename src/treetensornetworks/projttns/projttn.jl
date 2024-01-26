"""
ProjTTN
"""
struct ProjTTN{V} <: AbstractProjTTN{V}
  pos::Union{Vector{<:V},NamedEdge{V}} # TODO: cleanest way to specify effective Hamiltonian position?
  H::TTN{V}
  environments::Dictionary{NamedEdge{V},ITensor}
end

function ProjTTN(H::TTN)
  return ProjTTN(vertices(H), H, Dictionary{edgetype(H),ITensor}())
end

copy(P::ProjTTN) = ProjTTN(P.pos, copy(P.H), NamedGraphs.copy_keys_values(P.environments))

# trivial if we choose to specify position as above; only kept to allow using alongside
# ProjMPO
function set_nsite(P::ProjTTN, nsite)
  return P
end

function shift_position(P::ProjTTN, pos)
  return ProjTTN(pos, P.H, P.environments)
end

function make_environment!(P::ProjTTN{V}, psi::TTN{V}, e::NamedEdge{V})::ITensor where {V}
  # invalidate environment for opposite edge direction if necessary
  reverse(e) ∈ incident_edges(P) || unset!(P.environments, reverse(e))
  # do nothing if valid environment already present
  if haskey(P.environments, e)
    env = environment(P, e)
  else
    env = _compute_environment(P, psi, e)
    set!(P.environments, e, env)
  end
  @assert(
    hascommoninds(environment(P, e), psi[src(e)]),
    "Something went wrong, probably re-orthogonalized this edge in the same direction twice!"
  )
  return env
end

function make_environment(P::ProjTTN{V}, psi::TTN{V}, e::NamedEdge{V})::ITensor where {V}
  # invalidate environment for opposite edge direction if necessary
  #P=copy(P)
  reverse(e) ∈ incident_edges(P) || (P = invalidate_environment(P, reverse(e)))
  # do nothing if valid environment already present
  if haskey(P.environments, e)
    env = environment(P, e)
  else
    env = _compute_environment(P, psi, e)
  end
  @assert(
    hascommoninds(env, psi[src(e)]),
    "Something went wrong, probably re-orthogonalized this edge in the same direction twice!"
  )
  return env
end

function _compute_environment(
  P::ProjTTN{V}, psi::TTN{V}, e::NamedEdge{V}
)::ITensor where {V}
  if is_leaf(underlying_graph(P), src(e))
    # leaves are easy
    env = psi[src(e)] * P.H[src(e)] * dag(prime(psi[src(e)]))
  else
    # construct by contracting neighbors
    neighbor_envs = ITensor[]
    for n in setdiff(neighbors(underlying_graph(P), src(e)), [dst(e)])
      push!(neighbor_envs, make_environment!(P, psi, edgetype(P)(n, src(e))))
    end
    # manually heuristic for contraction order: two environments, site tensors, then
    # other environments
    frst, scnd, rst = _separate_first_two(neighbor_envs)
    itensor_map = vcat(psi[src(e)], frst, scnd, P.H[src(e)], dag(prime(psi[src(e)])), rst)
    # TODO: actually use optimal contraction sequence here
    env = reduce(*, itensor_map)
  end
  return env
end
