@eval module $(gensym())
using Graphs: vertices
using ITensorNetworks:
  ITensorNetworks,
  OpSum,
  ProjOuterProdTTN,
  ProjTTNSum,
  ttn,
  apply,
  contract,
  delta,
  dmrg,
  inner,
  mpo,
  random_mps,
  random_ttn,
  siteinds
using ITensorNetworks.ModelHamiltonians: ModelHamiltonians
using ITensors: prime, replaceinds, replaceprime
using ITensorMPS: ITensorMPS
using LinearAlgebra: norm, normalize
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using StableRNGs: StableRNG
using Test: @test, @test_broken, @testset

@testset "Contract MPO" begin
  N = 20
  s = siteinds("S=1/2", N)
  rng = StableRNG(1234)
  psi = random_mps(rng, s; link_space=8)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  for j in 1:(N - 2)
    os += 0.5, "S+", j, "S-", j + 2
    os += 0.5, "S-", j, "S+", j + 2
    os += "Sz", j, "Sz", j + 2
  end
  H = mpo(os, s)

  # Test basic usage with default parameters
  Hpsi = apply(H, psi; alg="fit", init=psi, nsweeps=1)
  @test inner(psi, Hpsi) ≈ inner(psi', H, psi) rtol = 1e-5
  # Test variational compression via DMRG
  Hfit = ProjOuterProdTTN(psi', H)
  e, Hpsi_via_dmrg = dmrg(Hfit, psi; updater_kwargs=(; which_eigval=:LR,), nsweeps=1)
  @test abs(inner(Hpsi_via_dmrg, Hpsi / norm(Hpsi))) ≈ 1 rtol = 1e-4
  # Test whether the interface works for ProjTTNSum with factors
  Hfit = ProjTTNSum([ProjOuterProdTTN(psi', H), ProjOuterProdTTN(psi', H)], [-0.2, -0.8])
  e, Hpsi_via_dmrg = dmrg(Hfit, psi; nsweeps=1, updater_kwargs=(; which_eigval=:SR,))
  @test abs(inner(Hpsi_via_dmrg, Hpsi / norm(Hpsi))) ≈ 1 rtol = 1e-4

  # Test basic usage for use with multiple ProjOuterProdTTN with default parameters
  # BLAS.axpy-like test
  os_id = OpSum()
  os_id += -1, "Id", 1, "Id", 2
  minus_identity = mpo(os_id, s)
  os_id = OpSum()
  os_id += +1, "Id", 1, "Id", 2
  identity = mpo(os_id, s)
  Hpsi = ITensorNetworks.sum_apply(
    [(H, psi), (minus_identity, psi)]; alg="fit", init=psi, nsweeps=3
  )
  @test inner(psi, Hpsi) ≈ (inner(psi', H, psi) - norm(psi)^2) rtol = 1e-5
  # Test the above via DMRG
  # ToDo: Investigate why this is broken
  Hfit = ProjTTNSum([ProjOuterProdTTN(psi', H), ProjOuterProdTTN(psi', identity)], [-1, 1])
  e, Hpsi_normalized = dmrg(Hfit, psi; nsweeps=3, updater_kwargs=(; which_eigval=:SR))
  @test_broken abs(inner(Hpsi, (Hpsi_normalized) / norm(Hpsi))) ≈ 1 rtol = 1e-5

  #
  # Change "top" indices of MPO to be a different set
  #
  t = siteinds("S=1/2", N)
  psit = deepcopy(psi)

  for j in 1:N
    H[j] *= delta(s[j]', t[j])
    psit[j] *= delta(s[j], t[j])
  end
  # Test with nsweeps=3
  Hpsi = contract(H, psi; alg="fit", init=psit, nsweeps=3)
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) rtol = 1e-5
  # Test with less good initial guess MPS not equal to psi
  psi_guess = truncate(psit; maxdim=2)
  Hpsi = contract(H, psi; alg="fit", nsweeps=4, init=psi_guess)
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) rtol = 1e-5

  # Test with nsite=1
  rng = StableRNG(1234)
  Hpsi_guess = random_mps(rng, t; link_space=32)
  Hpsi = contract(H, psi; alg="fit", init=Hpsi_guess, nsites=1, nsweeps=4)
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) rtol = 1e-4
end

@testset "Contract TTN" begin
  tooth_lengths = fill(4, 4)
  root_vertex = (1, 4)
  c = named_comb_tree(tooth_lengths)

  s = siteinds("S=1/2", c)
  rng = StableRNG(1234)
  psi = normalize(random_ttn(rng, s; link_space=8))

  os = ModelHamiltonians.heisenberg(c; J1=1, J2=1)
  H = ttn(os, s)

  # Test basic usage with default parameters
  Hpsi = apply(H, psi; alg="fit", init=psi, nsweeps=1, cutoff=eps())
  @test inner(psi, Hpsi) ≈ inner(psi', H, psi) rtol = 1e-5
  # Test usage with non-default parameters
  Hpsi = apply(
    H, psi; alg="fit", init=psi, nsweeps=5, maxdim=[16, 32], cutoff=[1e-4, 1e-8, 1e-12]
  )
  @test inner(psi, Hpsi) ≈ inner(psi', H, psi) rtol = 1e-2

  # Test basic usage for multiple ProjOuterProdTTN with default parameters
  # BLAS.axpy-like test
  os_id = OpSum()
  os_id += -1, "Id", first(vertices(s)), "Id", first(vertices(s))
  minus_identity = ttn(os_id, s)
  Hpsi = ITensorNetworks.sum_apply(
    [(H, psi), (minus_identity, psi)]; alg="fit", init=psi, nsweeps=1
  )
  @test inner(psi, Hpsi) ≈ (inner(psi', H, psi) - norm(psi)^2) rtol = 1e-5

  #
  # Change "top" indices of TTN to be a different set
  #
  t = siteinds("S=1/2", c)
  psit = deepcopy(psi)
  psit = replaceinds(psit, s => t)
  H = replaceinds(H, prime(s; links=[]) => t)

  # Test with nsweeps=2
  Hpsi = contract(H, psi; alg="fit", init=psit, nsweeps=2)
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) rtol = 1e-5

  # Test with less good initial guess MPS not equal to psi
  Hpsi_guess = truncate(psit; maxdim=2)
  Hpsi = contract(H, psi; alg="fit", nsweeps=4, init=Hpsi_guess)
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) rtol = 1e-5

  # Test with nsite=1
  rng = StableRNG(1234)
  Hpsi_guess = random_ttn(rng, t; link_space=32)
  Hpsi = contract(H, psi; alg="fit", nsites=1, nsweeps=10, init=Hpsi_guess)
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) rtol = 1e-2
end

@testset "Contract TTN with dangling inds" begin
  nbit = 3
  sites = siteinds("Qubit", nbit)

  # random_mpo does not support linkdims keyword.
  M1 = replaceprime(
    ITensorMPS.random_mpo(sites) + ITensorMPS.random_mpo(sites), 1 => 2, 0 => 1
  )
  M2 = ITensorMPS.random_mpo(sites) + ITensorMPS.random_mpo(sites)
  M12_ref = contract(M1, M2; alg="naive")
  t12_ref = ttn([M12_ref[v] for v in eachindex(M12_ref)])

  t1 = ttn([M1[v] for v in eachindex(M1)])
  t2 = ttn([M2[v] for v in eachindex(M2)])

  # Test with good initial guess
  @test contract(t1, t2; alg="fit", init=t12_ref, nsweeps=1) ≈ t12_ref rtol = 1e-7
end
end
