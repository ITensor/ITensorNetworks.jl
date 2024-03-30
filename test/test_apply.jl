using Test
using ITensorNetworks
using ITensorNetworks:
  environment, update, BeliefPropagationCache, VidalITensorNetwork, norm_sqr_network

using ITensors: inner, siteinds, op, apply
using SplitApplyCombine: group
using Random: seed!
using NamedGraphs: named_grid, PartitionVertex

@testset "apply" begin
  seed!(5623)
  g_dims = (2, 3)
  n = prod(g_dims)
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)
  χ = 2
  ψ = randomITensorNetwork(s; link_space=χ)
  v1, v2 = (2, 2), (1, 2)
  ψψ = norm_sqr_network(ψ)
  #Simple Belief Propagation Grouping
  bp_cache = BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
  bp_cache = update(bp_cache; maxiter=20)
  envsSBP = environment(bp_cache, PartitionVertex.([v1, v2]))

  ψv = VidalITensorNetwork(ψ)

  #This grouping will correspond to calculating the environments exactly (each column of the grid is a partition)
  bp_cache = BeliefPropagationCache(ψψ, group(v -> v[1][1], vertices(ψψ)))
  bp_cache = update(bp_cache; maxiter=20)
  envsGBP = environment(bp_cache, [(v1, "bra"), (v1, "ket"), (v2, "bra"), (v2, "ket")])

  inner_alg = "exact"

  ngates = 5

  for i in 1:ngates
    o = op("RandomUnitary", s[v1]..., s[v2]...)

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
    ψOv = apply(o, ψv; maxdim=χ, normalize=true)
    ψOVidal_symm = ITensorNetwork(ψOv)
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
      inner(ψOSBP, ψOexact; alg=inner_alg) /
      sqrt(inner(ψOexact, ψOexact; alg=inner_alg) * inner(ψOSBP, ψOSBP; alg=inner_alg))
    fVidal =
      inner(ψOVidal_symm, ψOexact; alg=inner_alg) / sqrt(
        inner(ψOexact, ψOexact; alg=inner_alg) *
        inner(ψOVidal_symm, ψOVidal_symm; alg=inner_alg),
      )
    fGBP =
      inner(ψOGBP, ψOexact; alg=inner_alg) /
      sqrt(inner(ψOexact, ψOexact; alg=inner_alg) * inner(ψOGBP, ψOGBP; alg=inner_alg))

    @test real(fGBP * conj(fGBP)) >= real(fSBP * conj(fSBP))

    @test isapprox(real(fSBP * conj(fSBP)), real(fVidal * conj(fVidal)); atol=1e-3)
  end
end
