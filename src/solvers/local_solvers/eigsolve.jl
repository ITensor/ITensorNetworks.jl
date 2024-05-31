using KrylovKit: eigsolve

function eigsolve_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  internal_kwargs,
  which_eigval=:SR,
  ishermitian=true,
  tol=1e-14,
  krylovdim=3,
  maxiter=1,
  verbosity=0,
  eager=false,
)
  howmany = 1
  vals, vecs, info = eigsolve(
    projected_operator![],
    init,
    howmany,
    which_eigval;
    ishermitian,
    tol,
    krylovdim,
    maxiter,
    verbosity,
    eager,
  )
  return vecs[1], (; info, eigvals=vals)
end

function bp_eigsolve_updater(init::ITensor, ∂ψAψ_bpc_∂rs::Vector, sqrt_mts, inv_sqrt_mts; kwargs...)

  #TODO: Put inv_sqrt_mts onto ∂ψAψ_bpc_∂r beforehand. Need to do this in an efficient way without
  #precontracting ∂ψAψ_bpc_∂r
  function get_new_state(∂ψAψ_bpc_∂rs::Vector, inv_sqrt_mts, state::ITensor; sequences = ["automatic" for i in length(∂ψAψ_bpc_∂rs)])
    state = noprime(contract([state; inv_sqrt_mts]))
    states = ITensor[dag(noprime(contract([copy(state); ∂ψAψ_bpc_∂r]; sequence = sequences[i]))) for (i, ∂ψAψ_bpc_∂r) in enumerate(∂ψAψ_bpc_∂rs)]
    state = reduce(+, states)
    return noprime(contract([state; (inv_sqrt_mts)]))
  end

  init = noprime(contract([init; sqrt_mts]))
  sequences = [optimal_contraction_sequence([init; ∂ψAψ_bpc_∂r]) for ∂ψAψ_bpc_∂r in ∂ψAψ_bpc_∂rs]
  get_new_state_partial = partial(get_new_state, ∂ψAψ_bpc_∂rs, inv_sqrt_mts; sequences)
  howmany = 1

  vals, vecs, info = eigsolve(get_new_state_partial,init,howmany,:SR; ishermitian = true, kwargs...)

  state = first(vecs)
  state = noprime(contract([state; inv_sqrt_mts]))

  return state, (; info, eigvals=vals)
end