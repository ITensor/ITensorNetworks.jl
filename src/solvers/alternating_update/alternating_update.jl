using ITensors: state
using ITensors.ITensorMPS: linkind
using NamedGraphs.GraphsExtensions: GraphsExtensions

function alternating_update(
  operator,
  init_state;
  nsweeps,  # define default for each solver implementation
  nsites, # define default for each level of solver implementation
  updater,  # this specifies the update performed locally
  outputlevel=default_outputlevel(),
  region_printer=nothing,
  sweep_printer=nothing,
  (sweep_observer!)=nothing,
  (region_observer!)=nothing,
  root_vertex=GraphsExtensions.default_root_vertex(init_state),
  extracter_kwargs=(;),
  extracter=default_extracter(),
  updater_kwargs=(;),
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
    operator,
    init_state,
    sweep_plans;
    outputlevel,
    sweep_observer!,
    region_observer!,
    sweep_printer,
    region_printer,
  )
end

function alternating_update(
  projected_operator,
  init_state,
  sweep_plans;
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
    update_observer!(
      sweep_observer!; state, which_sweep, sweep_time, outputlevel, sweep_plans
    )
    !isnothing(sweep_printer) &&
      sweep_printer(; state, which_sweep, sweep_time, outputlevel, sweep_plans)
    checkdone(;
      state,
      which_sweep,
      outputlevel,
      sweep_plan,
      sweep_plans,
      sweep_observer!,
      region_observer!,
    ) && break
  end
  return state
end

function alternating_update(operator::AbstractTTN, init_state::AbstractTTN; kwargs...)
  projected_operator = ProjTTN(operator)
  return alternating_update(projected_operator, init_state; kwargs...)
end

function alternating_update(operator::AbstractITensorNetwork, init_state::AbstractITensorNetwork, sweep_plans; kwargs...)
  ψOψ = QuadraticFormNetwork(operator, init_state)
  ψIψ = QuadraticFormNetwork(init_state)
  ψOψ_bpc = BeliefPropagationCache(ψOψ)
  ψIψ_bpc = BeliefPropagationCache(ψIψ)
  ψOψ_bpc = update(ψOψ_bpc)
  ψIψ_bpc = update(ψIψ_bpc)
  projected_operator = (ψOψ_bpc, ψIψ_bpc)
  return alternating_update(projected_operator, init_state, sweep_plans; kwargs...)
end

function alternating_update(
  operator::AbstractTTN, init_state::AbstractTTN, sweep_plans; kwargs...
)
  projected_operator = ProjTTN(operator)
  return alternating_update(projected_operator, init_state, sweep_plans; kwargs...)
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
  projected_operators = ProjTTNSum(operators)
  return alternating_update(projected_operators, init_state; kwargs...)
end

function alternating_update(
  operators::Vector{<:AbstractTTN}, init_state::AbstractTTN, sweep_plans; kwargs...
)
  projected_operators = ProjTTNSum(operators)
  return alternating_update(projected_operators, init_state, sweep_plans; kwargs...)
end
