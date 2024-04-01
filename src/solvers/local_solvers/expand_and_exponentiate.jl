function local_expand_and_exponentiate_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  region_kwargs,
  updater_kwargs,
)
  # expansion
  expand_kwargs = updater_kwargs.expand_kwargs  # defaults are set in the expansion_updater
  expanded_init, _ = two_site_expansion_updater(
    init;
    state!,
    projected_operator!,
    outputlevel,
    which_sweep,
    sweep_plan,
    which_region_update,
    region_kwargs,
    updater_kwargs=expand_kwargs,
  )

  # exponentiate
  # ToDo: also call exponentiate_updater, instead of reimplementing it here
  #
  default_exponentiate_kwargs = (;
    krylovdim=30,
    maxiter=100,
    verbosity=0,
    tol=1E-8,
    ishermitian=true,
    issymmetric=true,
    eager=true,
  )
  exponentiate_kwargs = updater_kwargs.exponentiate_kwargs
  exponentiate_kwargs = merge(default_exponentiate_kwargs, exponentiate_kwargs)  #last collection has precedence
  result, exp_info = exponentiate(
    projected_operator![], region_kwargs.time_step, expanded_init; exponentiate_kwargs...
  )

  # 
  #ToDo: return truncation error and append to info
  # truncate
  # ToDo: refactor
  # ToDo: either remove do_truncate, or make accessible as kwarg
  # ToDo: think about more elaborate ways to trigger truncation, i.e. truncate only once maxdim is hit?
  # ToDo: add truncation info
  do_truncate = true
  # simply return if we are not truncating 
  !do_truncate && return result, (; info=exp_info)
  # check for cases where we don't want to try truncating
  region = first(sweep_plan[which_region_update])
  typeof(region) <: NamedEdge && return result, (; info=exp_info)
  (; maxdim, cutoff) = region_kwargs
  region = only(region)
  next_region = if which_region_update == length(sweep_plan)
    nothing
  else
    first(sweep_plan[which_region_update + 1])
  end
  previous_region =
    which_region_update == 1 ? nothing : first(sweep_plan[which_region_update - 1])
  isnothing(next_region) && return result, (; info=exp_info)
  !(typeof(next_region) <: NamedEdge) && return result, (; info=exp_info)
  !(region == src(next_region) || region == dst(next_region)) &&
    return result, (; info=exp_info)
  # actually truncate
  left_inds = uniqueinds(state![], next_region)
  U, S, V = svd(
    result,
    left_inds;
    lefttags=tags(state![], next_region),
    righttags=tags(state![], next_region),
    maxdim,
    cutoff,
  )
  next_vertex = src(next_region) == region ? dst(next_region) : src(next_region)
  state = copy(state![])
  #@show inds(V)
  state[next_vertex] = state[next_vertex] * V
  _nsites = (region isa AbstractEdge) ? 0 : length(region) #should be 1
  #@show _nsites
  PH = copy(projected_operator![])
  PH = set_nsite(PH, 2)
  PH = position(PH, state, [region, next_vertex])
  PH = set_nsite(PH, _nsites)
  PH = position(PH, state, first(sweep_plan[which_region_update]))
  state![] = state
  projected_operator![] = PH
  return U * S, (; info=exp_info)
end
