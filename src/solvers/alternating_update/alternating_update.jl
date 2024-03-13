function alternating_update(
  projected_operator,
  init_state::AbstractTTN;
  sweep_plans,
  outputlevel=default_outputlevel(),
  checkdone=default_checkdone(),  # 
  (sweep_observer!)=nothing,
  sweep_printer=default_sweep_printer,#?
  (region_observer!)=nothing,
  region_printer=nothing,
)
  state = copy(init_state)
  @assert !isnothing(sweep_plans)
  for which_sweep in eachindex(sweep_plans)
    sweep_plan = sweep_plans[which_sweep]

    #ToDo: Hopefully not needed anymore, remove.
    sweep_plan = append_missing_namedtuple.(to_tuple.(sweep_plan))

    sweep_time = @elapsed begin
      for which_region_update in eachindex(sweep_plan)
        state, projected_operator = region_update(
          projected_operator,
          state;
          which_sweep,
          sweep_plan,
          region_printer,
          (region_observer!),
          which_region_update,
          outputlevel,  
        )
      end
    end

    update!(sweep_observer!; state, which_sweep, sweep_time, outputlevel, sweep_plans)
    !isnothing(sweep_printer) && sweep_printer(; state, which_sweep, sweep_time, outputlevel, sweep_plans)
    checkdone(; state, which_sweep, outputlevel, sweep_plan, sweep_plans, sweep_observer!,region_observer!) && break
  end
  return state
end

function alternating_update(operator::AbstractTTN, init_state::AbstractTTN; kwargs...)
  check_hascommoninds(siteinds, operator, init_state)
  check_hascommoninds(siteinds, operator, init_state')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  operator = ITensors.permute(operator, (linkind, siteinds, linkind))
  projected_operator = ProjTTN(operator)
  return alternating_update(projected_operator, init_state; kwargs...)
end

#ToDo: Fix docstring.
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
function alternating_update(
  operators::Vector{<:AbstractTTN}, init_state::AbstractTTN; kwargs...
)
  for operator in operators
    check_hascommoninds(siteinds, operator, init_state)
    check_hascommoninds(siteinds, operator, init_state')
  end
  operators .= ITensors.permute.(operators, Ref((linkind, siteinds, linkind)))
  projected_operators = ProjTTNSum(operators)
  return alternating_update(projected_operators, init_state; kwargs...)
end
