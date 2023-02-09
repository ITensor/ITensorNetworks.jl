function _compute_nsweeps(nsweeps, t, time_step)
  if isinf(t) && isnothing(nsweeps)
    nsweeps = 1
  elseif !isnothing(nsweeps) && time_step != t
    error("Cannot specify both nsweeps and a custom time_step in alternating_update")
  elseif isfinite(time_step) && abs(time_step) > 0.0 && isnothing(nsweeps)
    nsweeps = convert(Int, ceil(abs(t / time_step)))
    if !(nsweeps * time_step ≈ t)
      error("Time step $time_step not commensurate with total time t=$t")
    end
  end
  return nsweeps
end

function _extend_sweeps_param(param, nsweeps)
  if param isa Number
    eparam = fill(param, nsweeps)
  else
    length(param) == nsweeps && return param
    eparam = Vector(undef, nsweeps)
    eparam[1:length(param)] = param
    eparam[(length(param) + 1):end] .= param[end]
  end
  return eparam
end

function process_sweeps(;
  nsweeps=1,
  cutoff=fill(1E-16, nsweeps),
  maxdim=fill(typemax(Int), nsweeps),
  mindim=fill(1, nsweeps),
  noise=fill(0.0, nsweeps),
  kwargs...,
)
  maxdim = _extend_sweeps_param(maxdim, nsweeps)
  mindim = _extend_sweeps_param(mindim, nsweeps)
  cutoff = _extend_sweeps_param(cutoff, nsweeps)
  noise = _extend_sweeps_param(noise, nsweeps)
  return (; maxdim, mindim, cutoff, noise)
end

function alternating_update(
  solver,
  PH,
  psi0::AbstractTTN;
  checkdone=nothing,
  order=2,
  outputlevel=0,
  time::Number=Inf,
  time_start=0.0,
  time_step=time,
  nsweeps=nothing,
  write_when_maxdim_exceeds::Union{Int,Nothing}=nothing,
  kwargs...,
)
  nsweeps = _compute_nsweeps(nsweeps, time, time_step)
  maxdim, mindim, cutoff, noise = process_sweeps(; nsweeps, kwargs...)

  step_observer = get(kwargs, :step_observer!, nothing)

  tdvp_order = TDVPOrder(order, Base.Forward)

  psi = copy(psi0)

  # Keep track of the start of the current time step.
  # Helpful for tracking the total time, for example
  # when using time-dependent solvers.
  # This will be passed as a keyword argument to the
  # `solver`.
  current_time = time_start
  info = nothing
  for sw in 1:nsweeps
    if !isnothing(write_when_maxdim_exceeds) && maxdim[sw] > write_when_maxdim_exceeds
      if outputlevel >= 2
        println(
          "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw)), writing environment tensors to disk",
        )
      end
      PH = disk(PH)
    end

    sw_time = @elapsed begin
      psi, PH, info = update_step(
        tdvp_order,
        solver,
        PH,
        time_step,
        psi;
        kwargs...,
        current_time,
        outputlevel,
        sweep=sw,
        maxdim=maxdim[sw],
        mindim=mindim[sw],
        cutoff=cutoff[sw],
        noise=noise[sw],
      )
    end

    current_time += time_step

    update!(step_observer; psi, sweep=sw, outputlevel, current_time)

    if outputlevel >= 1
      print("After sweep ", sw, ":")
      print(" maxlinkdim=", maxlinkdim(psi))
      @printf(" maxerr=%.2E", info.maxtruncerr)
      print(" current_time=", round(current_time; digits=3))
      print(" time=", round(sw_time; digits=3))
      println()
      flush(stdout)
    end

    isdone = false
    if !isnothing(checkdone)
      isdone = checkdone(; psi, sweep=sw, outputlevel, kwargs...)
    end
    isdone && break
  end
  return psi
end

function alternating_update(solver, H::AbstractTTN, psi0::AbstractTTN; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = ITensors.permute(H, (linkind, siteinds, linkind))
  PH = ProjTTN(H)
  return alternating_update(solver, PH, psi0; kwargs...)
end

"""
    tdvp(Hs::Vector{MPO},psi0::MPS,t::Number; kwargs...)
    tdvp(Hs::Vector{MPO},psi0::MPS,t::Number, sweeps::Sweeps; kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to compute `exp(t*H)*psi0` using an efficient algorithm based
on alternating optimization of the MPS tensors and local Krylov
exponentiation of H.
                    
This version of `tdvp` accepts a representation of H as a
Vector of MPOs, Hs = [H1,H2,H3,...] such that H is defined
as H = H1+H2+H3+...
Note that this sum of MPOs is not actually computed; rather
the set of MPOs [H1,H2,H3,..] is efficiently looped over at 
each step of the algorithm when optimizing the MPS.

Returns:
* `psi::MPS` - time-evolved MPS
"""
function alternating_update(solver, Hs::Vector{<:AbstractTTN}, psi0::AbstractTTN; kwargs...)
  for H in Hs
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')
  end
  Hs .= ITensors.permute.(Hs, Ref((linkind, siteinds, linkind)))
  PHs = ProjTTNSum(Hs)
  return alternating_update(solver, PHs, psi0; kwargs...)
end
