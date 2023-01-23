using ITensors
using ITensorNetworks

include("05_utils.jl")

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
outputlevel = 1
nsteps = 10
time_steps = [n ≤ 2 ? -0.2im : -0.1im for n in 1:nsteps]

obs = Observer("times" => (; current_time) -> current_time, "psis" => (; psi) -> psi)

s = siteinds("S=1/2", N; conserve_qns=true)
H = MPO(heisenberg(N), s)

psi0 = MPS(s, n -> isodd(n) ? "Up" : "Dn")
psi = tdvp_nonuniform_timesteps(
  ProjMPO(H), psi0; time_steps, cutoff, outputlevel, (step_observer!)=obs
)

res = results(obs)
times = res["times"]
psis = res["psis"]

println("\nResults")
println("=======")
print("step = ", 0)
print(", time = ", zero(ComplexF64))
print(", ⟨Sᶻ⟩ = ", round(expect(psi0, "Sz"; vertices=[N ÷ 2]); digits=3))
println()
for n in 1:length(times)
  print("step = ", n)
  print(", time = ", round(times[n]; digits=3))
  print(", ⟨Sᶻ⟩ = ", round(expect(psis[n], "Sz"; vertices=[N ÷ 2]); digits=3))
  println()
end
