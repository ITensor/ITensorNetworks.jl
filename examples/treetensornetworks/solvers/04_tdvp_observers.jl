using ITensors
using ITensorNetworks
using Observers

function heisenberg(N)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  return os
end

N = 10
cutoff = 1e-12
tau = 0.1
ttotal = 1.0

s = siteinds("S=1/2", N; conserve_qns=true)
H = MPO(heisenberg(N), s)

function step(; sweep, bond, half_sweep)
  if bond == 1 && half_sweep == 2
    return sweep
  end
  return nothing
end

function current_time(; current_time, bond, half_sweep)
  if bond == 1 && half_sweep == 2
    return current_time
  end
  return nothing
end

function measure_sz(; psi, bond, half_sweep)
  if bond == 1 && half_sweep == 2
    return expect(psi, "Sz"; vertices=[N ÷ 2])
  end
  return nothing
end

function return_state(; psi, bond, half_sweep)
  if bond == 1 && half_sweep == 2
    return psi
  end
  return nothing
end

obs = Observer(
  "steps" => step, "times" => current_time, "psis" => return_state, "Sz" => measure_sz
)

psi = MPS(s, n -> isodd(n) ? "Up" : "Dn")
psi_f = tdvp(
  H,
  -im * ttotal,
  psi;
  time_step=-im * tau,
  cutoff,
  outputlevel=1,
  normalize=false,
  (observer!)=obs,
)

res = results(obs)
steps = res["steps"]
times = res["times"]
psis = res["psis"]
Sz = res["Sz"]

println("\nResults")
println("=======")
for n in 1:length(steps)
  print("step = ", steps[n])
  print(", time = ", round(times[n]; digits=3))
  print(", |⟨ψⁿ|ψⁱ⟩| = ", round(abs(inner(psis[n], psi)); digits=3))
  print(", |⟨ψⁿ|ψᶠ⟩| = ", round(abs(inner(psis[n], psi_f)); digits=3))
  print(", ⟨Sᶻ⟩ = ", round(Sz[n]; digits=3))
  println()
end
