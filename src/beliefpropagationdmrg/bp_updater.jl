using ITensors: contract
using ITensors.ContractionSequenceOptimization: optimal_contraction_sequence
using KrylovKit: eigsolve

default_krylov_kwargs() = (; tol = 1e-14, krylovdim = 20, maxiter = 3, verbosity = 0, eager = false, ishermitian = true)

#TODO: Put inv_sqrt_mts onto ∂ψOψ_bpc_∂r beforehand. Need to do this in an efficient way without
#precontracting ∂ψOψ_bpc_∂r and getting the index logic too messy
function get_new_state(∂ψOψ_bpc_∂rs::Vector, inv_sqrt_mts, sqrt_mts, state::ITensor; sequences = ["automatic" for i in length(∂ψOψ_bpc_∂rs)])
  state = noprime(contract([state; inv_sqrt_mts]))
  states = ITensor[dag(noprime(contract([copy(state); ∂ψOψ_bpc_∂r]; sequence))) for (∂ψOψ_bpc_∂r, sequence) in zip(∂ψOψ_bpc_∂rs, sequences)]
  state = reduce(+, states)
  return noprime(contract([state; (inv_sqrt_mts)]))
end

function bp_eigsolve_updater(init::ITensor, ∂ψOψ_bpc_∂rs::Vector, sqrt_mts, inv_sqrt_mts; krylov_kwargs = default_krylov_kwargs())
  
    init = noprime(contract([init; sqrt_mts]))
    sequences = [optimal_contraction_sequence([init; ∂ψOψ_bpc_∂r]) for ∂ψOψ_bpc_∂r in ∂ψOψ_bpc_∂rs]
    get_new_state_ = state -> get_new_state(∂ψOψ_bpc_∂rs, inv_sqrt_mts, sqrt_mts, state; sequences)
    howmany = 1
  
    vals, vecs, info = eigsolve(get_new_state_,init,howmany,:SR; krylov_kwargs...)
    state = noprime(contract([first(vecs); inv_sqrt_mts]))
  
    return state, (; info, eigvals=vals)
end