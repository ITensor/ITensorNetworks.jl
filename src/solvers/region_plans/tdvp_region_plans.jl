function tdvp_sub_time_steps(tdvp_order)
  if tdvp_order == 1
    return [1.0]
  elseif tdvp_order == 2
    return [1 / 2, 1 / 2]
  elseif tdvp_order == 4
    s = (2 - 2^(1 / 3))^(-1)
    return [s/2, s/2, 1/2 - s, 1/2 - s, s/2, s/2]
  else
    error("TDVP order of $tdvp_order not supported")
  end
end

function first_order_sweep(
  graph, time_step, dir=Base.Forward; updater_kwargs, nsites, kws...
)
  basic_fwd_sweep = post_order_dfs_plan(graph; nsites, kws...)
  updater_kwargs = (; nsites, time_step, updater_kwargs...)
  sweep = []
  for (j, (region, region_kws)) in enumerate(basic_fwd_sweep)
    push!(sweep, (region, (; nsites, updater_kwargs, region_kws...)))
    if length(region) == 2 && j < length(basic_fwd_sweep)
      rev_kwargs = (; updater_kwargs..., time_step=(-updater_kwargs.time_step))
      push!(sweep, ([last(region)], (; updater_kwargs=rev_kwargs, region_kws...)))
    end
  end
  if dir==Base.Reverse
    # Reverse regions as well as ordering of regions
    sweep = [(reverse(reg_kws[1]), reg_kws[2]) for reg_kws in reverse(sweep)]
  end
  return sweep
end

function tdvp_regions(graph, time_step; updater_kwargs, tdvp_order, nsites, kws...)
  sweep_plan = []
  for (step, weight) in enumerate(tdvp_sub_time_steps(tdvp_order))
    dir = isodd(step) ? Base.Forward : Base.Reverse
    append!(
      sweep_plan,
      first_order_sweep(graph, weight*time_step, dir; updater_kwargs, nsites, kws...),
    )
  end
  return sweep_plan
end
