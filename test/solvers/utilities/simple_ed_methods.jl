import ITensorNetworks as itn
import NetworkSolvers as ns

function ed_ground_state(H, psi0)
  ITensors.disable_warn_order()
  H = prod(H)
  psi = prod(psi0)
  # TODO: call KrylovKit instead
  expH = exp(H*(-20.0))
  for napply in 1:10
    psi = noprime(expH*psi)
    psi /= norm(psi)
  end
  E = scalar(prime(psi)*H*psi)
  return E, psi
end

function ed_time_evolution(
  H::itn.AbstractITensorNetwork,
  psi::itn.AbstractITensorNetwork,
  time_points;
  normalize=false,
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
    #psir, _ = ns.runge_kutta_solver(H_map,step,psi; order=4)
    #psix, _ = ns.exponentiate_solver(H_map,step,psi)
    #@show abs(scalar(dag(psi)*psix))
    #@show abs(scalar(dag(psir)*psix))
    if normalize
      psi /= norm(psi)
    end
  end
  return psi
end
