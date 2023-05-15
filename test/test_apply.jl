using ITensorNetworks
using ITensorNetworks:
  belief_propagation,
  get_environment,
  contract_inner,
  message_tensors,
  nested_graph_leaf_vertices,
  vidal_gauge,
  vidal_to_symmetric_gauge
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

  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  #Simple Belief Propagation Grouping
  vertex_groupsSBP = nested_graph_leaf_vertices(
    partition(partition(ψψ, group(v -> v[1], vertices(ψψ))); nvertices_per_partition=1)
  )
  Z = partition(ψψ; subgraph_vertices=vertex_groupsSBP)
  mtsSBP = message_tensors(Z)
  mtsSBP = belief_propagation(ψψ, mtsSBP; contract_kwargs=(; alg="exact"), niters=50)
  envsSBP = get_environment(ψψ, mtsSBP, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])

  ψ_vidal, bond_tensors = vidal_gauge(ψ, mtsSBP)

  #This grouping will correspond to calculating the environments exactly (each column of the grid is a partition)
  vertex_groupsGBP = nested_graph_leaf_vertices(
    partition(partition(ψψ, group(v -> v[1][1], vertices(ψψ))); nvertices_per_partition=1)
  )
  Z = partition(ψψ; subgraph_vertices=vertex_groupsSBP)
  mtsGBP = message_tensors(Z)
  mtsGBP = belief_propagation(ψψ, mtsGBP; contract_kwargs=(; alg="exact"), niters=50)
  envsGBP = get_environment(ψψ, mtsGBP, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])

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
    ψOVidal, bond_tensors_t = apply(o, ψ_vidal, bond_tensors; maxdim=χ, normalize=true)
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
