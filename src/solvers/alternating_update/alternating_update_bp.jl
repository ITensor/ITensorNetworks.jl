using ITensors: state
using ITensors.ITensorMPS: linkind
using NamedGraphs.GraphsExtensions: GraphsExtensions
using NamedGraphs.PartitionedGraphs: partitionvertices

default_vertex_sequence(ψ::AbstractITensorNetwork) =  vertices(ψ)

function alternating_update(
  alg::Algorithm"bp_onesite", A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork; start_vertex = GraphsExtensions.default_root_vertex(ψ), nsweeps = 1, nsites =1, kwargs...)

  ψAψ = QuadraticFormNetwork(A, ψ)
  ψIψ = QuadraticFormNetwork(ψ)
  ψAψ_bpc = BeliefPropagationCache(ψAψ)
  ψIψ_bpc = BeliefPropagationCache(ψIψ)

  sweep_plans = default_sweep_plans(nsweeps,ψ;root_vertex = start_vertex, nsites, extracter = default_extracter(), extracter_kwargs=(;), updater = eigsolve_updater, updater_kwargs=(;), transform_operator_kwargs=(;), inserter = default_inserter(), inserter_kwargs = (;), transform_operator=default_transform_operator())
  for which_sweep in eachindex(sweep_plans)
    sweep_plan = sweep_plans[which_sweep]
    for which_region_update in eachindex(sweep_plan)
      ψAψ_bpc = update(ψAψ_bpc)
      ψIψ_bpc = update(ψIψ_bpc)
      (region, region_kwargs) = sweep_plan[which_region_update]
      v = only(region)
      form_v = ket_vertex(ψAψ, v)
      ∂ψAψ_bpc_∂r = environment(ψAψ_bpc, [form_v])
      state = ψAψ[form_v]
      #Now get incoming messages to V
      messages = environment(ψIψ_bpc, partitionvertices(ψIψ_bpc, [form_v]))
      #Square root and put onto state

      #eigsolve

      #Inv square root onto new state

      #Update wavefuncion

    end
  end


  return 0
end