@eval module $(gensym())
using ITensorNetworks:
  ITensorNetwork,
  inner,
  inner_network,
  loginner,
  logscalar,
  random_tensornetwork,
  scalar,
  ttn,
  underlying_graph
using ITensorNetworks.ModelHamiltonians: heisenberg
using ITensors: dag, siteinds
using SplitApplyCombine: group
using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "Inner products, BP vs exact comparison" begin
  L = 4
  χ = 2
  g = NamedGraph(SimpleGraph(uniform_tree(L)))
  s = siteinds("S=1/2", g)
  rng = StableRNG(1234)
  y = random_tensornetwork(rng, s; link_space=χ)
  x = random_tensornetwork(rng, s; link_space=χ)

  #First lets do it with the flattened version of the network
  xy = inner_network(x, y)
  xy_scalar = scalar(xy)
  xy_scalar_bp = scalar(xy; alg="bp")
  xy_scalar_logbp = exp(logscalar(xy; alg="bp"))

  @test xy_scalar ≈ xy_scalar_bp
  @test xy_scalar_bp ≈ xy_scalar_logbp
  @test xy_scalar ≈ xy_scalar_logbp

  #Now lets do it via the inner function
  xy_scalar = inner(x, y; alg="exact")
  xy_scalar_bp = inner(x, y; alg="bp")
  xy_scalar_logbp = exp(loginner(x, y; alg="bp"))

  @test xy_scalar ≈ xy_scalar_bp
  @test xy_scalar_bp ≈ xy_scalar_logbp
  @test xy_scalar ≈ xy_scalar_logbp

  #test contraction of three layers for expectation values
  A = ITensorNetwork(ttn(heisenberg(g), s))
  xAy_scalar = inner(x, A, y; alg="exact")
  xAy_scalar_bp = inner(x, A, y; alg="bp")
  xAy_scalar_logbp = exp(loginner(x, A, y; alg="bp"))

  @test xAy_scalar ≈ xAy_scalar_bp
  @test xAy_scalar_bp ≈ xAy_scalar_logbp
  @test xAy_scalar ≈ xAy_scalar_logbp
end
end
