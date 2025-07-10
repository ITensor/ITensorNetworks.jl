function applyexp_sub_steps(order)
  if order == 1
    return [1.0]
  elseif order == 2
    return [1 / 2, 1 / 2]
  elseif order == 4
    s = (2 - 2^(1 / 3))^(-1)
    return [s/2, s/2, 1/2 - s, 1/2 - s, s/2, s/2]
  else
    error("Applyexp order of $order not supported")
  end
end

function first_order_sweep(
  graph, exponent_step, dir=Base.Forward; update_kwargs, nsites, kws...
)
  basic_fwd_sweep = post_order_dfs_plan(graph; nsites, kws...)
  update_kwargs = (; nsites, exponent_step, update_kwargs...)
  sweep = []
  for (j, (region, region_kws)) in enumerate(basic_fwd_sweep)
    push!(sweep, (region, (; nsites, update_kwargs, region_kws...)))
    if length(region) == 2 && j < length(basic_fwd_sweep)
      rev_kwargs = (; update_kwargs..., exponent_step=(-update_kwargs.exponent_step))
      push!(sweep, ([last(region)], (; update_kwargs=rev_kwargs, region_kws...)))
    end
  end
  if dir==Base.Reverse
    # Reverse regions as well as ordering of regions
    sweep = [(reverse(reg_kws[1]), reg_kws[2]) for reg_kws in reverse(sweep)]
  end
  return sweep
end

function applyexp_regions(graph, exponent_step; update_kwargs, order, nsites, kws...)
  sweep_plan = []
  for (step, weight) in enumerate(applyexp_sub_steps(order))
    dir = isodd(step) ? Base.Forward : Base.Reverse
    append!(
      sweep_plan,
      first_order_sweep(graph, weight*exponent_step, dir; update_kwargs, nsites, kws...),
    )
  end
  return sweep_plan
end
