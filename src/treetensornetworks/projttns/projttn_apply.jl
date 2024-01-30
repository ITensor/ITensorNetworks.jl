struct ProjTTNApply{V} <: AbstractProjTTN{V}
  pos::Union{Vector{<:V},NamedEdge{V}}
  init_state::TTN{V}
  operator::TTN{V}
  environments::Dictionary{NamedEdge{V},ITensor}
end

environments(p::ProjTTNApply) = p.environments
operator(p::ProjTTNApply) = p.operator
underlying_graph(p::ProjTTNApply) = underlying_graph(operator(p))
pos(p::ProjTTNApply) = p.pos
init_state(p::ProjTTNApply) = p.init_state

function ProjTTNApply(init_state::AbstractTTN, operator::AbstractTTN)
  return ProjTTNApply(
    vertextype(operator)[], init_state, operator, Dictionary{edgetype(operator),ITensor}()
  )
end

function copy(P::ProjTTNApply)
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

function set_environment!(p::ProjTTNApply, edge, env)
  set!(environments(p), edge, env)
  return p
end

function make_environment(
  P::ProjTTNApply, state::AbstractTTN, e::AbstractEdge
)
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
