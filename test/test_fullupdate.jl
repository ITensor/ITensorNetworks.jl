using ITensorNetworks
using ITensorNetworks:
  compute_message_tensors, get_environment, nested_graph_leaf_vertices, contract_inner
using Test
using Compat
using ITensors
using Metis
using NamedGraphs
using Random
using LinearAlgebra
using SplitApplyCombine

@testset "full_update" begin
  Random.seed!(5623)
  dims = (2, 3)
  n = prod(dims)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  χ = 2
  ψ = randomITensorNetwork(s; link_space=χ)
  v1, v2 = (2, 2), (1, 2)

  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  #Simple Belief Propagation Grouping
  vertex_groupsSBP = nested_graph_leaf_vertices(
    partition(partition(ψψ, group(v -> v[1], vertices(ψψ))); nvertices_per_partition=1)
  )
  mtsSBP = compute_message_tensors(ψψ; vertex_groups=vertex_groupsSBP)
  envsSBP = get_environment(Vector{ITensor}, ψψ, mtsSBP, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])

  #This grouping will correspond to calculating the environments exactly (each column of the grid is a partition)
  vertex_groupsGBP = nested_graph_leaf_vertices(
    partition(partition(ψψ, group(v -> v[1][1], vertices(ψψ))); nvertices_per_partition=1)
  )
  mtsGBP = compute_message_tensors(ψψ; vertex_groups=vertex_groupsGBP)
  envsGBP = get_environment(Vector{ITensor}, ψψ, mtsGBP, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])

  ngates = 5

  for i in 1:ngates
    o = ITensors.op("RandomUnitary", s[v1]..., s[v2]...)

    ψOexact = apply(o, ψ; maxdim=4 * χ)
    ψOSBP = apply(
      o,
      ψ;
      envs=envsSBP,
      maxdim=χ,
      normalize=true,
      print_fidelity_loss=true,
      envisposdef=true,
    )
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
    fGBP =
      contract_inner(ψOGBP, ψOexact) /
      sqrt(contract_inner(ψOexact, ψOexact) * contract_inner(ψOGBP, ψOGBP))

    @test real(fGBP * conj(fGBP)) >= real(fSBP * conj(fSBP))
  end
end
