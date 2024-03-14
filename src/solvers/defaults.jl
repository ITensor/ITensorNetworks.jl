default_outputlevel() = 0
default_nsites() = 2
default_nsweeps() = 1 #? or nothing?
default_extracter() = default_extracter
default_inserter() = default_inserter
default_checkdone() = (; kws...) -> false
default_transform_operator() = nothing
function default_region_printer(;
  cutoff,
  maxdim,
  mindim,
  outputlevel,
  state,
  sweep_plan,
  spec,
  which_region_update,
  which_sweep,
  kwargs...,
)
  if outputlevel >= 2
    region = first(sweep_plan[which_region_update])
    @printf("Sweep %d, region=%s \n", which_sweep, region)
    print("  Truncated using")
    @printf(" cutoff=%.1E", cutoff)
    @printf(" maxdim=%d", maxdim)
    @printf(" mindim=%d", mindim)
    println()
    if spec != nothing
      @printf(
        "  Trunc. err=%.2E, bond dimension %d\n",
        spec.truncerr,
        linkdim(state, edgetype(state)(region...))
      )
    end
    flush(stdout)
  end
end

#ToDo: Implement sweep_time_printer more generally
#ToDo: Implement more printers
#ToDo: Move to another file?
function default_sweep_time_printer(; outputlevel, which_sweep, kwargs...)
  if outputlevel >= 1
    sweeps_per_step = order ÷ 2
    if which_sweep % sweeps_per_step == 0
      current_time = (which_sweep / sweeps_per_step) * time_step
      println("Current time (sweep $which_sweep) = ", round(current_time; digits=3))
    end
  end
  return nothing
end

function default_sweep_printer(; outputlevel, state, which_sweep, sweep_time, kwargs...)
  if outputlevel >= 1
    print("After sweep ", which_sweep, ":")
    print(" maxlinkdim=", maxlinkdim(state))
    print(" cpu_time=", round(sweep_time; digits=3))
    println()
    flush(stdout)
  end
end

function default_alternating_updates(
  operator,
  init_state::AbstractTTN;
  nsweeps=default_nsweeps(),
  nsites=default_nsites(),
  outputlevel=default_outputlevel(),
  region_printer=nothing,
  sweep_printer=nothing,
  (sweep_observer!)=nothing,
  (region_observer!)=nothing,
  root_vertex=default_root_vertex(init_state),
  extracter_kwargs=(;),
  extracter=default_extracter(),
  updater_kwargs=(;),
  updater,  # this specifies the update performed locally
  inserter_kwargs=(;),
  inserter=default_inserter(),
  transform_operator_kwargs=(;),
  transform_operator=default_transform_operator(),
  kwargs...,
)
  inserter_kwargs = (; inserter_kwargs..., kwargs...)
  sweep_plans = default_sweep_plans(
    nsweeps,
    init_state;
    root_vertex,
    extracter,
    extracter_kwargs,
    updater,
    updater_kwargs,
    inserter,
    inserter_kwargs,
    transform_operator,
    transform_operator_kwargs,
    nsites,
  )
  return alternating_update(
    operator, init_state;
    outputlevel,
    sweep_plans,
    sweep_observer!,
    region_observer!,
    sweep_printer,
    region_printer,
  )
end
