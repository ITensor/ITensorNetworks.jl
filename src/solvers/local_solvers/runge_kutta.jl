
function runge_kutta_2(H, t, ψ0)
  Hψ = H(ψ0)
  H2ψ = H(Hψ)
  return (ψ0 + t * Hψ + (t^2 / 2) * H2ψ)
end

function runge_kutta_4(H, t, ψ0)
  k1 = H(ψ0)
  k2 = k1 + (t / 2) * H(k1)
  k3 = k1 + (t / 2) * H(k2)
  k4 = k1 + t * H(k3)
  return ψ0 + (t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end

function runge_kutta_solver(H, time, ψ; order=4, kws...)
  if order == 4
    Hψ = runge_kutta_4(H, time, ψ)
  elseif order == 2
    Hψ = runge_kutta_2(H, time, ψ)
  else
    error("For runge_kutta_solver, must specify `order` keyword")
  end
  return Hψ, (;)
end
