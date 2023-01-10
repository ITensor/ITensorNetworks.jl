using ITensors
using ITensorNetworks
using Observers
using Printf

using ITensorNetworks: tdvp_solver, tdvp_step, process_sweeps, TDVPOrder

function tdvp_nonuniform_timesteps(
  solver,
  PH,
  psi::MPS;
  time_steps,
  reverse_step=true,
  time_start=0.0,
  order=2,
  (step_observer!)=Observer(),
  kwargs...,
)
  nsweeps = length(time_steps)
  maxdim, mindim, cutoff, noise = process_sweeps(; nsweeps, kwargs...)
  tdvp_order = TDVPOrder(order, Base.Forward)
  current_time = time_start
  for sw in 1:nsweeps
    sw_time = @elapsed begin
      psi, PH, info = tdvp_step(
        tdvp_order,
        solver,
        PH,
        time_steps[sw],
        psi;
        kwargs...,
        current_time,
        reverse_step,
        sweep=sw,
        maxdim=maxdim[sw],
        mindim=mindim[sw],
        cutoff=cutoff[sw],
        noise=noise[sw],
      )
    end
    current_time += time_steps[sw]

    update!(step_observer!; psi, sweep=sw, outputlevel, current_time)

    if outputlevel ≥ 1
      print("After sweep ", sw, ":")
      print(" maxlinkdim=", maxlinkdim(psi))
      @printf(" maxerr=%.2E", info.maxtruncerr)
      print(" current_time=", round(current_time; digits=3))
      print(" time=", round(sw_time; digits=3))
      println()
      flush(stdout)
    end
  end
  return psi
end

function tdvp_nonuniform_timesteps(H, psi::MPS; kwargs...)
  return tdvp_nonuniform_timesteps(tdvp_solver(; kwargs...), H, psi; kwargs...)
end
