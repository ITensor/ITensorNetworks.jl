struct ProjTTNApply{V} <: AbstractProjTTN{V}
  pos::Union{Vector{<:V},NamedEdge{V}}
  init_state::AbstractTTN{V}
  operator::AbstractTTN{V}
  environments::Dictionary{NamedEdge{V},ITensor}
end

init_state(p::ProjTTNApply) = p.init_state

function ProjTTNApply(init_state::AbstractTTN, operator::AbstractTTN)
  return ProjTTNApply(
    vertextype(operator)[], init_state, operator, Dictionary{edgetype(operator),ITensor}()
  )
end

function copy(P::ProjTTNApply)
  return ProjTTNApply(
    pos(P), copy(init_state(P)), copy(operator(P)), copy_keys_values(environments(P))
  )
end

function unsafe_copy(P::ProjTTNApply)
  return ProjTTNApply(P.pos, copy(init_state(P)), copy(operator(P)), copy(environments(P)))
end
function set_nsite(P::ProjTTNApply, nsite)
  return P
end

function shift_position(P::ProjTTNApply, pos)
  return ProjTTNApply(pos, init_state(P), operator(P), environments(P))
end

function set_environments(p::ProjTTNApply, environments)
  return ProjTTNApply(pos(p), init_state(p), operator(p), environments)
end

function set_environment(p::ProjTTNApply, edge, env)
  newenv = merge(p.environments, Dictionary((edge,), (env,)))
  return ProjTTNApply(pos(p), init_state(p), operator(p), newenv)
end

set_environment!(p::ProjTTNApply, edge, env) = set!(environments(P), edge, env)

function make_environment!(
  P::ProjTTNApply{V}, state::AbstractTTN{V}, e::NamedEdge{V}
)::ITensor where {V}
  # invalidate environment for opposite edge direction if necessary
  reverse(e) ∈ incident_edges(P) || unset!(environments(P), reverse(e))
  # do nothing if valid environment already present
  if haskey(environments(P), e)
    env = environment(P, e)
  else
    if is_leaf(underlying_graph(P), src(e))
      # leaves are easy
      env = init_state(P)[src(e)] * operator(P)[src(e)] * dag(state[src(e)])
    else
      # construct by contracting neighbors
      neighbor_envs = ITensor[]
      for n in setdiff(neighbors(underlying_graph(P), src(e)), [dst(e)])
        push!(neighbor_envs, make_environment!(P, state, edgetype(P)(n, src(e))))
      end
      # manually heuristic for contraction order: two environments, site tensors, then
      # other environments
      frst, scnd, rst = _separate_first_two(neighbor_envs)
      itensor_map = vcat(
        init_state(P)[src(e)], frst, scnd, operator(P)[src(e)], dag(state[src(e)]), rst
      ) # no prime here in comparison to the same routine for Projttn
      # TODO: actually use optimal contraction sequence here
      env = reduce(*, itensor_map)
    end
    # cache
    set_environment!(P, e, env)
  end
  @assert(
    hascommoninds(environment(P, e), state[src(e)]),
    "Something went wrong, probably re-orthogonalized this edge in the same direction twice!"
  )
  return env
end

function make_environment(
  P::ProjTTNApply{V}, state::AbstractTTN{V}, e::NamedEdge{V}
)::ProjTTNApply{V} where {V}
  # invalidate environment for opposite edge direction if necessary
  reverse(e) ∈ incident_edges(P) || (P = invalidate_environment(P, reverse(e)))
  # do nothing if valid environment already present
  if !haskey(environments(P), e)
    if is_leaf(underlying_graph(P), src(e))
      # leaves are easy
      env = init_state(P)[src(e)] * operator(P)[src(e)] * dag(state[src(e)])
    else
      # construct by contracting neighbors
      neighbor_envs = ITensor[]
      for n in setdiff(neighbors(underlying_graph(P), src(e)), [dst(e)])
        P = make_environment(P, state, edgetype(P)(n, src(e)))
        push!(neighbor_envs, environment(P, edgetype(P)(n, src(e))))
      end
      # manually heuristic for contraction order: two environments, site tensors, then
      # other environments
      frst, scnd, rst = _separate_first_two(neighbor_envs)
      itensor_map = vcat(
        init_state(P)[src(e)], frst, scnd, operator(P)[src(e)], dag(state[src(e)]), rst
      ) # no prime here in comparison to the same routine for Projttn
      # TODO: actually use optimal contraction sequence here
      env = reduce(*, itensor_map)
    end
    P = set_environment(P, e, env)
  end
  @assert(
    hascommoninds(environment(P, e), state[src(e)]),
    "Something went wrong, probably re-orthogonalized this edge in the same direction twice!"
  )
  return P
end
