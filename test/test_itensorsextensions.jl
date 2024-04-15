@eval module $(gensym())
using ITensors:
  ITensors,
  ITensor,
  Index,
  QN,
  dag,
  delta,
  inds,
  noprime,
  prime,
  randomITensor,
  replaceind,
  replaceinds,
  sim
using ITensorNetworks.ITensorsExtensions: map_eigvals
using Random: Random
using Test: @test, @testset

Random.seed!(1234)
@testset "ITensorsExtensions" begin
  @testset "Test map eigvals without QNS" begin
    for eltype in [Float64, ComplexF64]
      for n in [2, 3, 5, 10]
        i, j = Index(n, "i"), Index(n, "j")
        linds, rinds = Index[i], Index[j]
        A = randn(eltype, (n, n))
        A = A * A'
        P = ITensor(A, i, j)
        sqrtP = map_eigvals(sqrt, P, linds, rinds; ishermitian=true)
        inv_P = dag(map_eigvals(inv, P, linds, rinds; ishermitian=true))
        inv_sqrtP = dag(map_eigvals(inv ∘ sqrt, P, linds, rinds; ishermitian=true))

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

  @testset "Test map eigvals with QNS" begin
    for eltype in [Float64, ComplexF64]
      for n in [2, 3, 5, 10]
        i, j = Index.(([QN() => n], [QN() => n]))
        A = randomITensor(eltype, i, j)
        P = A * prime(dag(A), i)
        sqrtP = map_eigvals(sqrt, P, i, i'; ishermitian=true)
        inv_P = dag(map_eigvals(inv, P, i, i'; ishermitian=true))
        inv_sqrtP = dag(map_eigvals(inv ∘ sqrt, P, i, i'; ishermitian=true))

        new_ind = noprime(sim(i'))
        sqrtPdag = replaceind(dag(sqrtP), i', new_ind)
        P2 = replaceind(sqrtP * sqrtPdag, new_ind, i)
        @test P2 ≈ P atol = 1e-12

        inv_P = replaceind(inv_P, i', new_ind)
        I = replaceind(inv_P * P, new_ind, i)
        @test I ≈ op("I", i) atol = 1e-12

        inv_sqrtP = replaceind(inv_sqrtP, i', new_ind)
        I = replaceind(inv_sqrtP * sqrtP, new_ind, i)
        @test I ≈ op("I", i) atol = 1e-12
      end
    end
  end
end
