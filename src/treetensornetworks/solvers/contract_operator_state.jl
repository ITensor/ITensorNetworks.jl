function contract_solver(; kwargs...)
  function solver(PH, t, psi; kws...)
    v = ITensor(1.0)
    for j in sites(PH)
      v *= PH.psi0[j]
    end
    Hpsi0 = contract(PH, v)
    return Hpsi0, nothing
  end
  return solver
end

function ITensors.contract(
  ::ITensors.Algorithm"fit",
  A::IsTreeOperator,
  psi0::ST;
  init_state=psi0,
  nsweeps=1,
  kwargs...,
)::ST where {ST<:IsTreeState}
  n = nv(A)
  n != nv(psi0) && throw(
    DimensionMismatch("Number of sites operator ($n) and state ($(nv(psi0))) do not match"),
  )
  if n == 1
    v = only(vertices(psi0))
    return ST([A[v] * psi0[v]])
  end

  check_hascommoninds(siteinds, A, psi0)

  # In case A and psi0 have the same link indices
  A = sim(linkinds, A)

  # Fix site and link inds of init_state
  init_state = deepcopy(init_state)
  init_state = sim(linkinds, init_state)
  for v in vertices(psi0)
    replaceinds!(
      init_state[v], siteinds(init_state, v), uniqueinds(siteinds(A, v), siteinds(psi0, v))
    )
  end

  t = Inf
  reverse_step = false
  PH = proj_operator_apply(psi0, A)
  psi = tdvp(
    contract_solver(; kwargs...), PH, t, init_state; nsweeps, reverse_step, kwargs...
  )

  return psi
end

# extra ITensors overloads for tree tensor networks
function ITensors.contract(A::TTNO, ψ::TTNS; alg="fit", kwargs...)
  return contract(ITensors.Algorithm(alg), A, ψ; kwargs...)
end

function ITensors.apply(A::TTNO, ψ::TTNS; kwargs...)
  Aψ = contract(A, ψ; kwargs...)
  return replaceprime(Aψ, 1 => 0)
end
