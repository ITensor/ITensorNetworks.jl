function _tdvp_compute_nsweeps(t; kwargs...)
  time_step::Number = get(kwargs, :time_step, t)
  nsweeps::Union{Int,Nothing} = get(kwargs, :nsweeps, nothing)
  if isinf(t) && isnothing(nsweeps)
    nsweeps = 1
  elseif !isnothing(nsweeps) && time_step != t
    error("Cannot specify both time_step and nsweeps in tdvp")
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

function process_sweeps(; kwargs...)
  nsweeps = get(kwargs, :nsweeps, 1)
  maxdim = get(kwargs, :maxdim, fill(typemax(Int), nsweeps))
  mindim = get(kwargs, :mindim, fill(1, nsweeps))
  cutoff = get(kwargs, :cutoff, fill(1E-16, nsweeps))
  noise = get(kwargs, :noise, fill(0.0, nsweeps))

  maxdim = _extend_sweeps_param(maxdim, nsweeps)
  mindim = _extend_sweeps_param(mindim, nsweeps)
  cutoff = _extend_sweeps_param(cutoff, nsweeps)
  noise = _extend_sweeps_param(noise, nsweeps)

  return (; maxdim, mindim, cutoff, noise)
end

function tdvp(solver, PH, t::Number, psi0::AbstractTTN; kwargs...)
  reverse_step = get(kwargs, :reverse_step, true)

  nsweeps = _tdvp_compute_nsweeps(t; kwargs...)
  maxdim, mindim, cutoff, noise = process_sweeps(; nsweeps, kwargs...)

  time_start::Number = get(kwargs, :time_start, 0.0)
  time_step::Number = get(kwargs, :time_step, t)
  order = get(kwargs, :order, 2)
  tdvp_order = TDVPOrder(order, Base.Forward)

  checkdone = get(kwargs, :checkdone, nothing)
  write_when_maxdim_exceeds::Union{Int,Nothing} = get(
    kwargs, :write_when_maxdim_exceeds, nothing
  )
  observer = get(kwargs, :observer!, nothing)
  step_observer = get(kwargs, :step_observer!, nothing)
  outputlevel::Int = get(kwargs, :outputlevel, 0)

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
      psi, PH, info = tdvp_step(
        tdvp_order,
        solver,
        PH,
        time_step,
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

"""
    tdvp(H::MPS,psi0::MPO,t::Number; kwargs...)
    tdvp(H::TTN,psi0::TTN,t::Number; kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to compute `exp(t*H)*psi0` using an efficient algorithm based
on alternating optimization of the state tensors and local Krylov
exponentiation of H.
                    
Returns:
* `psi` - time-evolved state

Optional keyword arguments:
* `outputlevel::Int = 1` - larger outputlevel values resulting in printing more information and 0 means no output
* `observer` - object implementing the [Observer](@ref observer) interface which can perform measurements and stop early
* `write_when_maxdim_exceeds::Int` - when the allowed maxdim exceeds this value, begin saving tensors to disk to free memory in large calculations
"""
function tdvp(solver, H::AbstractTTN, t::Number, psi0::AbstractTTN; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = ITensors.permute(H, (linkind, siteinds, linkind))
  PH = ProjTTN(H)
  return tdvp(solver, PH, t, psi0; kwargs...)
end

function tdvp(solver, t::Number, H, psi0::AbstractTTN; kwargs...)
  return tdvp(solver, H, t, psi0; kwargs...)
end

function tdvp(solver, H, psi0::AbstractTTN, t::Number; kwargs...)
  return tdvp(solver, H, t, psi0; kwargs...)
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
function tdvp(solver, Hs::Vector{<:AbstractTTN}, t::Number, psi0::AbstractTTN; kwargs...)
  for H in Hs
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')
  end
  Hs .= ITensors.permute.(Hs, Ref((linkind, siteinds, linkind)))
  PHs = ProjTTNSum(Hs)
  return tdvp(solver, PHs, t, psi0; kwargs...)
end
