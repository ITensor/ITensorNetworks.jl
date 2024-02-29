
function alternating_update(
  projected_operator,
  init_state::AbstractTTN;
  sweep_plans,  #this is really the only one beig pass all the way down
  outputlevel, # we probably want to extract this one indeed for passing to observer etc.
  checkdone=(; kws...) -> false,  ### move outside
  (sweep_observer!)=observer(),
  sweep_printer=sweep_printer,
  write_when_maxdim_exceeds::Union{Int,Nothing}=nothing, ### move outside
)
  state = copy(init_state)
  for (which_sweep, sweep_plan) in enumerate(sweep_plans)
    if !isnothing(write_when_maxdim_exceeds) && #fix passing this
      maxdim[which_sweep] > write_when_maxdim_exceeds
      if outputlevel >= 2
        println(
          "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim[which_sweep] = $(maxdim[which_sweep]), writing environment tensors to disk",
        )
      end
      projected_operator = disk(projected_operator)
    end
    sw_time = @elapsed begin
      state, projected_operator = sweep_update(
        projected_operator, state; outputlevel, which_sweep, sweep_plan
      )
    end

    update!(sweep_observer!; state, which_sweep, sw_time, outputlevel)
    sweep_printer(; state, which_sweep, sw_time, outputlevel)
    checkdone(; state, which_sweep, outputlevel, sweep_plan) && break
  end
  return state
end

function alternating_update(H::AbstractTTN, init_state::AbstractTTN; kwargs...)
  check_hascommoninds(siteinds, H, init_state)
  check_hascommoninds(siteinds, H, init_state')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = ITensors.permute(H, (linkind, siteinds, linkind))
  projected_operator = ProjTTN(H)
  return alternating_update(projected_operator, init_state; kwargs...)
end

"""
    tdvp(Hs::Vector{MPO},init_state::MPS,t::Number; kwargs...)
    tdvp(Hs::Vector{MPO},init_state::MPS,t::Number, sweeps::Sweeps; kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to compute `exp(t*H)*init_state` using an efficient algorithm based
on alternating optimization of the MPS tensors and local Krylov
exponentiation of H.
                    
This version of `tdvp` accepts a representation of H as a
Vector of MPOs, Hs = [H1,H2,H3,...] such that H is defined
as H = H1+H2+H3+...
Note that this sum of MPOs is not actually computed; rather
the set of MPOs [H1,H2,H3,..] is efficiently looped over at 
each step of the algorithm when optimizing the MPS.

Returns:
* `state::MPS` - time-evolved MPS
"""
function alternating_update(Hs::Vector{<:AbstractTTN}, init_state::AbstractTTN; kwargs...)
  for H in Hs
    check_hascommoninds(siteinds, H, init_state)
    check_hascommoninds(siteinds, H, init_state')
  end
  Hs .= ITensors.permute.(Hs, Ref((linkind, siteinds, linkind)))
  projected_operators = ProjTTNSum(Hs)
  return alternating_update(projected_operators, init_state; kwargs...)
end
