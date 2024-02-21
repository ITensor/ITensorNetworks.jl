using ITensorNetworks
using ITensorNetworks:
  incoming_messages,
  update,
  contract_inner,
  vidal_gauge,
  vidal_apply,
  vidal_to_symmetric_gauge,
  norm_network
using Test
using Compat
using ITensors
using Metis
using NamedGraphs
using Random
using LinearAlgebra
using SplitApplyCombine

@testset "apply" begin
  Random.seed!(5623)
  g_dims = (2, 3)
  n = prod(g_dims)
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)
  χ = 2
  ψ = randomITensorNetwork(s; link_space=χ)
  v1, v2 = (2, 2), (1, 2)
  ψψ = norm_network(ψ)

  #Simple Belief Propagation Grouping
  bp_cache = BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
  bp_cache = update(bp_cache; maxiters=20)
  envsSBP = incoming_messages(bp_cache, PartitionVertex.([v1, v2]))

  ψ_vidal, bond_tensors = vidal_gauge(ψ; (cache!)=Ref(bp_cache))

  #This grouping will correspond to calculating the environments exactly (each column of the grid is a partition)
  bp_cache = BeliefPropagationCache(ψψ, group(v -> v[1][1], vertices(ψψ)))
  bp_cache = update(bp_cache; maxiters=20)
  envsGBP = incoming_messages(bp_cache, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])

  ngates = 5

  for i in 1:ngates
    o = ITensors.op("RandomUnitary", s[v1]..., s[v2]...)

    ψOexact = apply(o, ψ; cutoff=1e-16)
    ψOSBP = apply(
      o,
      ψ;
      envs=envsSBP,
      maxdim=χ,
      normalize=true,
      print_fidelity_loss=true,
      envisposdef=true,
    )
    ψOVidal, bond_tensors_t = vidal_apply(
      o, ψ_vidal, bond_tensors; maxdim=χ, normalize=true
    )
    ψOVidal_symm, _ = vidal_to_symmetric_gauge(ψOVidal, bond_tensors_t)
    ψOGBP = apply(
      o,
      ψ;
      envs=envsGBP,
      maxdim=χ,
      normalize=true,
      print_fidelity_loss=true,
      envisposdef=true,
    )
    fSBP =
      contract_inner(ψOSBP, ψOexact) /
      sqrt(contract_inner(ψOexact, ψOexact) * contract_inner(ψOSBP, ψOSBP))
    fVidal =
      contract_inner(ψOVidal_symm, ψOexact) /
      sqrt(contract_inner(ψOexact, ψOexact) * contract_inner(ψOVidal_symm, ψOVidal_symm))
    fGBP =
      contract_inner(ψOGBP, ψOexact) /
      sqrt(contract_inner(ψOexact, ψOexact) * contract_inner(ψOGBP, ψOGBP))

    @test real(fGBP * conj(fGBP)) >= real(fSBP * conj(fSBP))

    @test isapprox(real(fSBP * conj(fSBP)), real(fVidal * conj(fVidal)); atol=1e-3)
  end
end
