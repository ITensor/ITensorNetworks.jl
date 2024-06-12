using NamedGraphs.GraphsExtensions: is_tree
using NamedGraphs.PartitionedGraphs: partitionvertices, partitionedges, PartitionEdge
using ITensorNetworks:
  ITensorNetwork,
  QuadraticFormNetwork,
  BeliefPropagationCache,
  update,
  default_message_update,
  delete_messages
using ITensors: scalar

include("utils.jl")
include("bp_extracter.jl")
include("bp_inserter.jl")
include("bp_updater.jl")
include("graphsextensions.jl")

default_bp_update_kwargs(ψ::ITensorNetwork) = is_tree(ψ) ? (;) : (; maxiter=20, tol=1e-6)

function initialize_caches(
  ψ_init::ITensorNetwork,
  operators::Vector{ITensorNetwork};
  cache_update_kwargs=default_bp_update_kwargs(ψ_init),
)
  ψ = copy(ψ_init)
  ψIψ = QuadraticFormNetwork(ψ)
  ψIψ_bpc = BeliefPropagationCache(ψIψ)

  ψOψs = QuadraticFormNetwork[QuadraticFormNetwork(operator, ψ) for operator in operators]
  ψOψ_bpcs = BeliefPropagationCache[BeliefPropagationCache(ψOψ) for ψOψ in ψOψs]
  return (ψ, ψOψ_bpcs, ψIψ_bpc)
end

function bp_dmrg(
  ψ_init::ITensorNetwork,
  operators::Vector{<:ITensorNetwork};
  nsites=1,
  no_sweeps=1,
  bp_update_kwargs=default_bp_update_kwargs(ψ_init),
  energy_calc_fun,
)
  L = length(vertices(ψ_init))
  state, ψOψ_bpcs, ψIψ_bpc = initialize_caches(ψ_init, operators)
  state_vertices = collect(vertices(state))
  regions = [[v] for v in vcat(state_vertices, reverse(state_vertices))]

  state, ψOψ_bpcs, ψIψ_bpc = renormalize_update_norm_cache(
    state, ψOψ_bpcs, ψIψ_bpc; cache_update_kwargs=bp_update_kwargs
  )

  energy = real(energy_calc_fun(state, ψIψ_bpc))
  println("Initial energy density is $(energy)")
  energies = [energy]

  for i in 1:no_sweeps
    println("Beginning sweep $i")
    for region in regions
      println("Updating vertex $region")

      local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts = bp_extracter(
        state, ψOψ_bpcs, ψIψ_bpc, region
      )

      local_state, energy = bp_eigsolve_updater(
        local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts
      )

      state, ψOψ_bpcs, ψIψ_bpc = bp_inserter(
        state, ψOψ_bpcs, ψIψ_bpc, local_state, region; bp_update_kwargs
      )

      energy = energy_calc_fun(state, ψIψ_bpc)
      append!(energies, energy)
      println("Current energy density is $(energy)")
    end
  end

  return state, energies
end
