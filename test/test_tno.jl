@eval module $(gensym())
using Graphs: vertices
using ITensorNetworks:
  apply,
  flatten_networks,
  group_commuting_itensors,
  gate_group_to_tno,
  get_tnos,
  random_tensornetwork,
  siteinds
using ITensors: ITensor, inner, noprime
using ITensorNetworks.ModelHamiltonians: ModelHamiltonians
using NamedGraphs: named_grid
using Test: @test, @testset

@testset "TN operator Basics" begin
  L = 3
  g = named_grid((L, L))
  s = siteinds("S=1/2", g)

  ℋ = ModelHamiltonians.ising(g; h=1.5)
  gates = Vector{ITensor}(ℋ, s)
  gate_groups = group_commuting_itensors(gates)

  @test typeof(gate_groups) == Vector{Vector{ITensor}}

  #Construct a number of tnos whose product is prod(gates)
  tnos = get_tnos(s, gates)
  @test length(tnos) == length(gate_groups)

  #Construct a single tno which represents prod(gates)
  single_tno = gate_group_to_tno(s, gates)

  ψ = random_tensornetwork(s; link_space=2)

  ψ_gated = copy(ψ)
  for gate in gates
    ψ_gated = apply(gate, ψ_gated)
  end
  ψ_tnod = copy(ψ)
  for tno in tnos
    ψ_tnod = flatten_networks(ψ_tnod, tno)
    for v in vertices(ψ_tnod)
      ψ_tnod[v] = noprime(ψ_tnod[v])
    end
  end
  ψ_tno = copy(ψ)
  ψ_tno = flatten_networks(ψ_tno, single_tno)
  for v in vertices(ψ_tno)
    ψ_tno[v] = noprime(ψ_tno[v])
  end

  z1 = inner(ψ_gated, ψ_gated)
  z2 = inner(ψ_tnod, ψ_tnod)
  z3 = inner(ψ_tno, ψ_tno)
  f12 = inner(ψ_tnod, ψ_gated) / sqrt(z1 * z2)
  f13 = inner(ψ_tno, ψ_gated) / sqrt(z1 * z3)
  f23 = inner(ψ_tno, ψ_tnod) / sqrt(z2 * z3)
  @test f12 * conj(f12) ≈ 1.0
  @test f13 * conj(f13) ≈ 1.0
  @test f23 * conj(f23) ≈ 1.0
end
end
