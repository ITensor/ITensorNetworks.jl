function _compute_nsweeps(nsteps, t, time_step, order)
  nsweeps_per_step = order / 2
  nsweeps = 1
  if !isnothing(nsteps) && time_step != t
    error("Cannot specify both nsteps and time_step in tdvp")
  elseif isfinite(time_step) && abs(time_step) > 0.0 && isnothing(nsteps)
    nsweeps = convert(Int, nsweeps_per_step * ceil(abs(t / time_step)))
    if !(nsweeps / nsweeps_per_step * time_step ≈ t)
      println(
        "Time that will be reached = nsweeps/nsweeps_per_step * time_step = ",
        nsweeps / nsweeps_per_step * time_step,
      )
      println("Requested total time t = ", t)
      error("Time step $time_step not commensurate with total time t=$t")
    end
  end
  return nsweeps
end

function sub_time_steps(order)
  if order == 1
    return [1.0]
  elseif order == 2
    return [1 / 2, 1 / 2]
  elseif order == 4
    s = 1.0 / (2 - 2^(1 / 3))
    return [s / 2, s / 2, (1 - 2 * s) / 2, (1 - 2 * s) / 2, s / 2, s / 2]
  else
    error("Trotter order of $order not supported")
  end
end

function tdvp_sweep(
  order::Int,
  nsite::Int,
  time_step::Number,
  graph::AbstractGraph;
  root_vertex=default_root_vertex(graph),
  reverse_step=true,
)
  sweep = []
  for (substep, fac) in enumerate(sub_time_steps(order))
    sub_time_step = time_step * fac
    half = half_sweep(
      direction(substep),
      graph,
      make_region;
      root_vertex,
      nsite,
      region_args=(; substep, time_step=sub_time_step),
      reverse_args=(; substep, time_step=-sub_time_step),
      reverse_step,
    )
    append!(sweep, half)
  end
  return sweep
end

function tdvp(
  updater,
  H,
  t::Number,
  init::AbstractTTN;
  time_step::Number=t,
  nsite=2,
  nsteps=nothing,
  order::Integer=2,
  (sweep_observer!)=observer(),
  root_vertex=default_root_vertex(init),
  reverse_step=true,
  updater_kwargs=NamedTuple(;),
  kwargs...,
)
  nsweeps = _compute_nsweeps(nsteps, t, time_step, order)
  region_updates = tdvp_sweep(order, nsite, time_step, init; root_vertex, reverse_step)

  function sweep_time_printer(; outputlevel, which_sweep, kwargs...)
    if outputlevel >= 1
      sweeps_per_step = order ÷ 2
      if sweep % sweeps_per_step == 0
        current_time = (which_sweep / sweeps_per_step) * time_step
        println("Current time (sweep $which_sweep) = ", round(current_time; digits=3))
      end
    end
    return nothing
  end

  insert_function!(sweep_observer!, "sweep_time_printer" => sweep_time_printer)

  psi = alternating_update(
    updater, H, init; nsweeps, sweep_observer!, region_updates, updater_kwargs, kwargs...
  )

  # remove sweep_time_printer from sweep_observer!
  select!(sweep_observer!, Observers.DataFrames.Not("sweep_time_printer"))

  return psi
end

"""
    tdvp(H::TTN, t::Number, psi0::TTN; kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to approximately compute `exp(H*t)*psi0` using an efficient algorithm based
on alternating optimization of the state tensors and local Krylov
exponentiation of H. The time parameter `t` can be a real or complex number.
                    
Returns:
* `psi` - time-evolved state

Optional keyword arguments:
* `time_step::Number = t` - time step to use when evolving the state. Smaller time steps generally give more accurate results but can make the algorithm take more computational time to run.
* `nsteps::Integer` - evolve by the requested total time `t` by performing `nsteps` of the TDVP algorithm. More steps can result in more accurate results but require more computational time to run. (Note that only one of the `time_step` or `nsteps` parameters can be provided, not both.)
* `outputlevel::Int = 1` - larger outputlevel values resulting in printing more information and 0 means no output
* `observer` - object implementing the Observer interface which can perform measurements and stop early
* `write_when_maxdim_exceeds::Int` - when the allowed maxdim exceeds this value, begin saving tensors to disk to free memory in large calculations
"""
function tdvp(H, t::Number, init::AbstractTTN; updater=exponentiate_updater, kwargs...)
  return tdvp(updater, H, t, init; kwargs...)
end
