import ITensorNetworks: AbstractITensorNetwork

function ed_ground_state(H, psi0)
  ITensors.disable_warn_order()
  H = prod(H)
  psi = prod(psi0)
  expH = exp(H*(-20.0))
  for napply in 1:10
    psi = noprime(expH*psi)
    psi /= norm(psi)
  end
  E = scalar(prime(psi)*H*psi)
  return E, psi
end

function ed_time_evolution(
  H::AbstractITensorNetwork, psi::AbstractITensorNetwork, time_points; normalize=false
)
  ITensors.disable_warn_order()
  H = prod(H)
  psi = prod(psi)
  exponents = [-im*t for t in time_points]
  steps = diff([0.0, exponents...])[2:end]
  H_map = ψ -> noprime(H*psi)
  for step in steps
    expH = exp(H * step)
    psi = noprime(expH * psi)
    if normalize
      psi /= norm(psi)
    end
  end
  return psi
end
