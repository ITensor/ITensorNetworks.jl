@eval module $(gensym())
using ITensors:
  ITensors,
  ITensor,
  Index,
  QN,
  apply,
  dag,
  delta,
  inds,
  mapprime,
  noprime,
  norm,
  op,
  permute,
  prime,
  random_itensor,
  replaceind,
  replaceinds,
  sim,
  swapprime
using ITensorNetworks.ITensorsExtensions: eigendecomp, map_eigvals
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

  @testset "Fermionic eigendecomp" begin
    s1 = Index([QN("Nf", 0, -1)=>2, QN("Nf", 1, -1)=>2], "Site,Fermion,n=1")
    s2 = Index([QN("Nf", 0, -1)=>2, QN("Nf", 1, -1)=>2], "Site,Fermion,n=2")

    # Make a random Hermitian matrix-like 4th order ITensor
    T = random_itensor(s1', s2', dag(s2), dag(s1))
    T = apply(T, swapprime(dag(T), 0=>1))
    @test T ≈ swapprime(dag(T), 0=>1) # check Hermitian

    Ul, D, Ur = eigendecomp(T, [s1', s2'], [dag(s1), dag(s2)]; ishermitian=true)

    @test Ul*D*Ur ≈ T
  end

  @testset "Fermionic map eigvals tests" begin
    s1 = Index([QN("Nf", 0, -1)=>2, QN("Nf", 1, -1)=>2], "Site,Fermion,n=1")
    s2 = Index([QN("Nf", 0, -1)=>2, QN("Nf", 1, -1)=>2], "Site,Fermion,n=2")

    # Make a random Hermitian matrix ITensor
    M = random_itensor(s1', dag(s1))
    #M = mapprime(prime(M)*swapprime(dag(M),0=>1),2=>1)
    M = apply(M, swapprime(dag(M), 0=>1))

    # Make a random Hermitian matrix-like 4th order ITensor
    T = random_itensor(s1', s2', dag(s2), dag(s1))
    T = apply(T, swapprime(dag(T), 0=>1))

    # Matrix test
    sqrtM = map_eigvals(sqrt, M, s1', dag(s1); ishermitian=true)
    @test M ≈ apply(sqrtM, sqrtM)

    ## Tensor test
    sqrtT = map_eigvals(sqrt, T, [s1', s2'], [dag(s1), dag(s2)]; ishermitian=true)
    @test T ≈ apply(sqrtT, sqrtT)

    # Permute and test again
    T = permute(T, dag(s2), s2', dag(s1), s1')
    sqrtT = map_eigvals(sqrt, T, [s1', s2'], [dag(s1), dag(s2)]; ishermitian=true)
    @test T ≈ apply(sqrtT, sqrtT)

    ## Explicitly passing indices in different, valid orders
    sqrtT = map_eigvals(sqrt, T, [s2', s1'], [dag(s2), dag(s1)]; ishermitian=true)
    @test T ≈ apply(sqrtT, sqrtT)
    sqrtT = map_eigvals(sqrt, T, [dag(s2), dag(s1)], [s2', s1'], ; ishermitian=true)
    @test T ≈ apply(sqrtT, sqrtT)
    sqrtT = map_eigvals(sqrt, T, [dag(s1), dag(s2)], [s1', s2'], ; ishermitian=true)
    @test T ≈ apply(sqrtT, sqrtT)
  end
end
end
