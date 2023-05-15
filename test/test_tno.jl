using Test
using ITensorNetworks
using ITensors
using Random

using ITensorNetworks: gate_group_to_tno, get_tnos, group_gates, contract_inner

@testset "TN operator Basics" begin

  L = 3
  g = named_grid((L, L))
  s = siteinds("S=1/2", g)

  ℋ = ising(g; h = 1.5)
  gates = Vector{ITensor}(ℋ, s)
  gate_groups = group_gates(s, gates)

  @test typeof(gate_groups) == Vector{Vector{ITensor}}
  tnos = get_tnos(s, gates)
  @test length(tnos) == length(gate_groups)

  ψ = randomITensorNetwork(s; link_space = 2)
  ψ_gated = copy(ψ)
  for gate in gates
    ψ_gated = apply(gate, ψ_gated)
  end
  ψ_tnod = copy(ψ)
  for tno in tnos
    ψ_tnod = flatten_networks(ψ_tnod, tno)
    for v in vertices(ψ_tnod)
        noprime!(ψ_tnod[v])
    end
  end

  z1 = contract_inner(ψ_gated, ψ_gated )
  z2 = contract_inner(ψ_tnod, ψ_tnod)
  f = contract_inner(ψ_tnod, ψ_gated) / sqrt(z1*z2)
  @show f*conj(f) ≈ 1.0



end
