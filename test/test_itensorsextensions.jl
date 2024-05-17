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
  op,
  prime,
  random_itensor,
  replaceind,
  replaceinds,
  sim
using ITensorNetworks.ITensorsExtensions: map_eigvals
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "ITensorsExtensions" begin
  @testset "Test map eigvals without QNS (eltype=$elt, dim=$n)" for elt in (
      Float32, Float64, Complex{Float32}, Complex{Float64}
    ),
    n in (2, 3, 5, 10)

    i, j = Index(n, "i"), Index(n, "j")
    linds, rinds = Index[i], Index[j]
    rng = StableRNG(1234)
    A = randn(rng, elt, (n, n))
    A = A * A'
    P = ITensor(A, i, j)
    sqrtP = map_eigvals(sqrt, P, linds, rinds; ishermitian=true)
    inv_P = dag(map_eigvals(inv, P, linds, rinds; ishermitian=true))
    inv_sqrtP = dag(map_eigvals(inv ∘ sqrt, P, linds, rinds; ishermitian=true))

    sqrtPdag = replaceind(dag(sqrtP), i, i')
    P2 = replaceind(sqrtP * sqrtPdag, i', j)
    @test P2 ≈ P

    invP = replaceind(inv_P, i, i')
    I = invP * P
    @test I ≈ delta(elt, inds(I))

    inv_sqrtP = replaceind(inv_sqrtP, i, i')
    I = inv_sqrtP * sqrtP
    @test I ≈ delta(elt, inds(I))
  end

  @testset "Test map eigvals with QNS (eltype=$elt, dim=$n)" for elt in (
      Float32, Float64, Complex{Float32}, Complex{Float64}
    ),
    n in (2, 3, 5, 10)

    i, j = Index.(([QN() => n], [QN() => n]))
    rng = StableRNG(1234)
    A = random_itensor(rng, elt, i, j)
    P = A * prime(dag(A), i)
    sqrtP = map_eigvals(sqrt, P, i, i'; ishermitian=true)
    inv_P = dag(map_eigvals(inv, P, i, i'; ishermitian=true))
    inv_sqrtP = dag(map_eigvals(inv ∘ sqrt, P, i, i'; ishermitian=true))

    new_ind = noprime(sim(i'))
    sqrtPdag = replaceind(dag(sqrtP), i', new_ind)
    P2 = replaceind(sqrtP * sqrtPdag, new_ind, i)
    @test P2 ≈ P

    inv_P = replaceind(inv_P, i', new_ind)
    I = replaceind(inv_P * P, new_ind, i)
    @test I ≈ op("I", i)

    inv_sqrtP = replaceind(inv_sqrtP, i', new_ind)
    I = replaceind(inv_sqrtP * sqrtP, new_ind, i)
    @test I ≈ op("I", i)
  end
end
end
