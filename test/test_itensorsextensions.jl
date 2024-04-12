using ITensors:
  ITensors,
  ITensor,
  Index,
  randomITensor,
  dag,
  replaceind,
  noprime,
  prime,
  inds,
  delta,
  QN,
  denseblocks,
  replaceinds,
  dir,
  array
using ITensorNetworks.ITensorsExtensions: map_itensor
using ITensorNetworks: siteinds, random_tensornetwork
using NamedGraphs: named_grid
using Random
using Test: @test, @testset
using LinearAlgebra: ishermitian, isposdef

Random.seed!(1234)
@testset "ITensorsExtensions" begin
  for eltype in [Float64, ComplexF64]
    for n in [2, 3, 5, 10]
      i, j = Index(n, "i"), Index(n, "j")
      A = randn(eltype, (n, n))
      A = A * A'
      P = ITensor(A, i, j)
      sqrtP = map_itensor(sqrt, P)
      inv_P = dag(map_itensor(inv, P))
      inv_sqrtP = dag(map_itensor(inv ∘ sqrt, P))

      sqrtPdag = replaceind(dag(sqrtP), i, i')
      P2 = replaceind(sqrtP * sqrtPdag, i', j)
      @test P2 ≈ P atol = 1e-12

      invP = replaceind(inv_P, i, i')
      I = invP * P
      @test I ≈ delta(eltype, inds(I)) atol = 1e-12

      inv_sqrtP = replaceind(inv_sqrtP, i, i')
      I = inv_sqrtP * sqrtP
      @test I ≈ delta(eltype, inds(I)) atol = 1e-12
    end
  end
end
