using ITensors
using ITensorNetworks

n = 10
s = siteinds("S=1/2", n)

function heisenberg(n)
  os = OpSum()
  for j in 1:(n - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  return os
end

H = MPO(heisenberg(n), s)
ψ = randomMPS(s, "↑"; linkdims=10)

@show inner(ψ', H, ψ) / inner(ψ, ψ)

ϕ = tdvp(
  H,
  -1.0,
  ψ;
  nsweeps=20,
  reverse_step=false,
  normalize=true,
  maxdim=30,
  cutoff=1e-10,
  outputlevel=1,
)

@show inner(ϕ', H, ϕ) / inner(ϕ, ϕ)

e2, ϕ2 = dmrg(H, ψ; nsweeps=10, maxdim=20, cutoff=1e-10)

@show inner(ϕ2', H, ϕ2) / inner(ϕ2, ϕ2)

ϕ3 = ITensorNetworks.dmrg(H, ψ; nsweeps=10, maxdim=20, cutoff=1e-10, outputlevel=1)

@show inner(ϕ3', H, ϕ3) / inner(ϕ3, ϕ3)
