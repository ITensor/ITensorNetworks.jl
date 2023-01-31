struct ProjTTNApply{V} <: AbstractProjTTN{V}
  pos::Union{Vector{<:V},NamedEdge{V}}
  psi0::TTN{V}
  H::TTN{V}
  environments::Dictionary{NamedEdge{V},ITensor}
end

function ProjTTNApply(psi0::TTN, H::TTN)
  return ProjTTNApply(vertextype(H)[], psi0, H, Dictionary{edgetype(H),ITensor}())
end

function copy(P::ProjTTNApply)
  return ProjTTNApply(P.pos, copy(P.psi0), copy(P.H), copy(P.environments))
end

function set_nsite(P::ProjTTNApply, nsite)
  return P
end

function shift_position(P::ProjTTNApply, pos)
  return ProjTTNApply(pos, P.psi0, P.H, P.environments)
end

function make_environment!(
  P::ProjTTNApply{V}, psi::TTN{V}, e::NamedEdge{V}
)::ITensor where {V}
  # invalidate environment for opposite edge direction if necessary
  reverse(e) ∈ incident_edges(P) || unset!(P.environments, reverse(e))
  # do nothing if valid environment already present
  if haskey(P.environments, e)
    env = environment(P, e)
  else
    if is_leaf(underlying_graph(P), src(e))
      # leaves are easy
      env = P.psi0[src(e)] * P.H[src(e)] * dag(psi[src(e)])
    else
      # construct by contracting neighbors
      neighbor_envs = ITensor[]
      for n in setdiff(neighbors(underlying_graph(P), src(e)), [dst(e)])
        push!(neighbor_envs, make_environment!(P, psi, edgetype(P)(n, src(e))))
      end
      # manually heuristic for contraction order: two environments, site tensors, then
      # other environments
      frst, scnd, rst = _separate_first_two(neighbor_envs)
      itensor_map = vcat(P.psi0[src(e)], frst, scnd, P.H[src(e)], dag(psi[src(e)]), rst)
      # TODO: actually use optimal contraction sequence here
      env = reduce(*, itensor_map)
    end
    # cache
    set!(P.environments, e, env)
  end
  @assert(
    hascommoninds(environment(P, e), psi[src(e)]),
    "Something went wrong, probably re-orthogonalized this edge in the same direction twice!"
  )
  return env
end
