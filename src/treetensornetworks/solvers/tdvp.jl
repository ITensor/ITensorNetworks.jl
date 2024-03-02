default_outputlevel() = 0

#ToDo: Cleanup _compute_nsweeps, maybe restrict flexibility to simplify code
function _compute_nsweeps(nsweeps::Int, t::Number, time_step::Number)
  return error("Cannot specify both nsweeps and time_step in tdvp")
end

function _compute_nsweeps(nsweeps::Nothing, t::Number, time_step::Number)
  @assert isfinite(time_step) && abs(time_step) > 0.0
  nsweeps = convert(Int, nsweeps_per_step * ceil(abs(t / time_step)))
  if !(nsweeps / nsweeps_per_step * time_step ≈ t)
    println(
      "Time that will be reached = nsweeps/nsweeps_per_step * time_step = ",
      nsweeps / nsweeps_per_step * time_step,
    )
    println("Requested total time t = ", t)
    error("Time step $time_step not commensurate with total time t=$t")
  end
  return nsweeps, extend(time_step, nsweeps)
end

function _compute_nsweeps(nsweeps::Int, t::Number, time_step::Nothing)
  time_step = extend(t / nsweeps, nsweeps)
  return nsweeps, time_step
end

function _compute_nsweeps(nsweeps, t::Number, time_step::Vector)
  diff_time = t - sum(time_step)
  if diff_time < eps()
    if length(time_step) != nsweeps
      error(
        "A vector of timesteps that sums up to total time t=$t was supplied,
  but its length (=$(length(time_step))) does not agree with supplied number of sweeps (=$(nsweeps)).",
      )
    end
    return time_step, nsweeps
  end
  if isnothing(nsweeps)
    #extend time_step to reach final time t
    last_time_step = last(time_step)
    nsweepstopad = ceil(abs(diff_time / last_time_step))
    if !(sum(time_step) + nsweepstopad * last_time_step ≈ t)
      println(
        "Time that will be reached = nsweeps/nsweeps_per_step * time_step = ",
        nsweeps / nsweeps_per_step * time_step,
      )
      println("Requested total time t = ", t)
      error("Time step $time_step not commensurate with total time t=$t")
    end
    time_step = extend(time_step, length(time_step) + nsweepstopad)
    nsweeps = length(time_step)
  else
    nsweepstopad = nsweeps - length(time_step)
    remaining_time_step = diff_time / nsweepstopad
    append!(time_step, extend(remaining_time_step, nsweepstopad))
  end
  return time_step, nsweeps
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

"""
    tdvp(operator::TTN, t::Number, init_state::TTN; kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to approximately compute `exp(operator*t)*init_state` using an efficient algorithm based
on alternating optimization of the state tensors and local Krylov
exponentiation of operator. The time parameter `t` can be a real or complex number.
                    
Returns:
* `state` - time-evolved state

Optional keyword arguments:
* `time_step::Number = t` - time step to use when evolving the state. Smaller time steps generally give more accurate results but can make the algorithm take more computational time to run.
* `nsteps::Integer` - evolve by the requested total time `t` by performing `nsteps` of the TDVP algorithm. More steps can result in more accurate results but require more computational time to run. (Note that only one of the `time_step` or `nsteps` parameters can be provided, not both.)
* `outputlevel::Int = 1` - larger outputlevel values resulting in printing more information and 0 means no output
* `observer` - object implementing the Observer interface which can perform measurements and stop early
* `write_when_maxdim_exceeds::Int` - when the allowed maxdim exceeds this value, begin saving tensors to disk to free memory in large calculations
"""
function tdvp(
  operator,
  t::Number,
  init_state::AbstractTTN;
  time_step::Number=nothing,
  nsites=2,
  nsweeps=nothing,
  order::Integer=2,
  outputlevel=default_outputlevel,
  region_printer=nothing,
  sweep_printer=nothing,
  (sweep_observer!)=nothing,
  (region_observer!)=nothing,
  root_vertex=default_root_vertex(init_state),
  reverse_step=true,
  extracter_kwargs=(;),
  extracter=default_extractor(), # ToDo: extracter could be inside extracter_kwargs, at the cost of having to extract it in region_update
  updater_kwargs=(;),
  updater=exponentiate_updater,
  inserter_kwargs=(;),
  inserter=default_inserter(),
  kwargs...,
)
  # move slurped kwargs into inserter
  inserter_kwargs = (; inserter_kwargs..., kwargs...) # slurp unbound kwargs into inserter

  sweep_plans = tdvp_sweep_plans(
    nsteps,
    t,
    time_step,
    order,
    nsites,
    init_state;
    root_vertex,
    reverse_step,
    extracter,
    extracter_kwargs,
    updater,
    updater_kwargs,
    inserter,
    inserter_kwargs,
  )

  #=
  function sweep_time_printer(; outputlevel, which_sweep, kwargs...)
    if outputlevel >= 1
      sweeps_per_step = order ÷ 2
      if which_sweep % sweeps_per_step == 0
        current_time = (which_sweep / sweeps_per_step) * time_step
        println("Current time (sweep $which_sweep) = ", round(current_time; digits=3))
      end
    end
    return nothing
  end
  =#
  #insert_function!(sweep_observer!, "sweep_time_printer" => sweep_time_printer)

  state = alternating_update(
    operator, init_state; outputlevel, sweep_observer!, sweep_plans
  )

  # remove sweep_time_printer from sweep_observer!
  #select!(sweep_observer!, Observers.DataFrames.Not("sweep_time_printer"))

  return state
end
