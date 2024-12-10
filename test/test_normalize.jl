@eval module $(gensym())
using ITensorNetworks:
  BeliefPropagationCache,
  QuadraticFormNetwork,
  edge_scalars,
  norm_sqr_network,
  random_tensornetwork,
  vertex_scalars,
  rescale
using ITensors: dag, inner, siteinds, scalar
using Graphs: SimpleGraph, uniform_tree
using LinearAlgebra: normalize
using NamedGraphs: NamedGraph
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "Normalize" begin

  #First lets do a flat tree 
  nx, ny = 2, 3
  χ = 2
  rng = StableRNG(1234)

  g = named_comb_tree((nx, ny))
  tn = random_tensornetwork(rng, g; link_space=χ)

  tn_r = rescale(tn; alg="exact")
  @test scalar(tn_r; alg="exact") ≈ 1.0

  tn_r = rescale(tn; alg="bp")
  @test scalar(tn_r; alg="exact") ≈ 1.0

  #Now a state on a loopy graph
  Lx, Ly = 3, 2
  χ = 2
  rng = StableRNG(1234)

  g = named_grid((Lx, Ly))
  s = siteinds("S=1/2", g)
  x = random_tensornetwork(rng, s; link_space=χ)

  ψ = normalize(x; alg="exact")
  @test scalar(norm_sqr_network(ψ); alg="exact") ≈ 1.0

  ψIψ_bpc = Ref(BeliefPropagationCache(QuadraticFormNetwork(x)))
  ψ = normalize(x; alg="bp", (cache!)=ψIψ_bpc, update_cache=true)
  ψIψ_bpc = ψIψ_bpc[]
  @test all(x -> x ≈ 1.0, edge_scalars(ψIψ_bpc))
  @test all(x -> x ≈ 1.0, vertex_scalars(ψIψ_bpc))
  @test scalar(QuadraticFormNetwork(ψ); alg="bp") ≈ 1.0
end
end
