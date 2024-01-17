
function _extend_sweeps_param(param, nsweeps)
  if param isa Number
    eparam = fill(param, nsweeps)
  else
    length(param) >= nsweeps && return param[1:nsweeps]
    eparam = Vector(undef, nsweeps)
    eparam[1:length(param)] = param
    eparam[(length(param) + 1):end] .= param[end]
  end
  return eparam
end

function process_sweeps(
  nsweeps;
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
  return maxdim, mindim, cutoff, noise, kwargs
end

function sweep_printer(; outputlevel, psi, which_sweep, sw_time)
  if outputlevel >= 1
    print("After sweep ", which_sweep, ":")
    print(" maxlinkdim=", maxlinkdim(psi))
    print(" cpu_time=", round(sw_time; digits=3))
    println()
    flush(stdout)
  end
end

function alternating_update(
  updater,
  PH,
  psi0::AbstractTTN;
  checkdone=(; kws...) -> false,
  outputlevel::Integer=0,
  nsweeps::Integer=1,
  (sweep_observer!)=observer(),
  sweep_printer=sweep_printer,
  write_when_maxdim_exceeds::Union{Int,Nothing}=nothing,
  updater_kwargs,
  kwargs...,
)
  maxdim, mindim, cutoff, noise, kwargs = process_sweeps(nsweeps; kwargs...)

  psi = copy(psi0)

  insert_function!(sweep_observer!, "sweep_printer" => sweep_printer) # FIX THIS

  for which_sweep in 1:nsweeps
    if !isnothing(write_when_maxdim_exceeds) &&
      maxdim[which_sweep] > write_when_maxdim_exceeds
      if outputlevel >= 2
        println(
          "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim[which_sweep] = $(maxdim[which_sweep]), writing environment tensors to disk",
        )
      end
      PH = disk(PH)
    end

    sw_time = @elapsed begin
      psi, PH = sweep_update(
        updater,
        PH,
        psi;
        outputlevel,
        which_sweep,
        sweep_params=(;
          maxdim=maxdim[which_sweep],
          mindim=mindim[which_sweep],
          cutoff=cutoff[which_sweep],
          noise=noise[which_sweep],
        ),
        updater_kwargs,
        kwargs...,
      )
    end

    update!(sweep_observer!; psi, which_sweep, sw_time, outputlevel)

    checkdone(; psi, which_sweep, outputlevel, kwargs...) && break
  end
  select!(sweep_observer!, Observers.DataFrames.Not("sweep_printer"))
  return psi
end

function alternating_update(updater, H::AbstractTTN, psi0::AbstractTTN; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = ITensors.permute(H, (linkind, siteinds, linkind))
  PH = ProjTTN(H)
  return alternating_update(updater, PH, psi0; kwargs...)
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
function alternating_update(
  updater, Hs::Vector{<:AbstractTTN}, psi0::AbstractTTN; kwargs...
)
  for H in Hs
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')
  end
  Hs .= ITensors.permute.(Hs, Ref((linkind, siteinds, linkind)))
  PHs = ProjTTNSum(Hs)
  return alternating_update(updater, PHs, psi0; kwargs...)
end
