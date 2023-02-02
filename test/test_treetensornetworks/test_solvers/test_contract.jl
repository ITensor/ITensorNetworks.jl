using ITensors
using ITensorNetworks
using Random
using Test

@testset "Contract MPO" begin
  N = 20
  s = siteinds("S=1/2", N)
  psi = random_mps(s; internal_inds_space=8)

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
  Hpsi = apply(H, psi; alg="fit", init=psi')
  @test inner(psi, Hpsi) ≈ inner(psi', H, psi) atol = 1E-5

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
  Hpsi = apply(H, psi; alg="fit", nsweeps=3)
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) atol = 1E-5

  # Test with less good initial guess MPS not equal to psi
  psi_guess = truncate(psi; maxdim=2)
  Hpsi = apply(H, psi; alg="fit", nsweeps=4, init_state=psi_guess)
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) atol = 1E-5

  # Test with nsite=1
  Hpsi_guess = random_mps(t; internal_inds_space=32)
  Hpsi = apply(H, psi; alg="fit", init=Hpsi_guess, nsite=1, nsweeps=4)
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) atol = 1E-4
end

@testset "Contract TTN" begin
  tooth_lengths = fill(2, 3)
  root_vertex = (3, 2)
  c = named_comb_tree(tooth_lengths)

  s = siteinds("S=1/2", c)
  psi = normalize!(random_ttn(s; link_space=8))

  os = ITensorNetworks.heisenberg(c; J1=1, J2=1)
  H = TTN(os, s)

  # Test basic usage with default parameters
  Hpsi = apply(H, psi; alg="fit")
  @test inner(psi, Hpsi) ≈ inner(psi', H, psi) atol = 1E-5

  #
  # Change "top" indices of TTN to be a different set
  #
  t = siteinds("S=1/2", c)
  psit = deepcopy(psi)
  psit = replaceinds(psit, s => t)
  H = replaceinds(H, prime(s; links=[]) => t)

  # Test with nsweeps=2
  Hpsi = apply(H, psi; alg="fit", nsweeps=2)
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) atol = 1E-5

  # Test with less good initial guess MPS not equal to psi
  Hpsi_guess = truncate(psit; maxdim=2)
  Hpsi = apply(H, psi; alg="fit", nsweeps=4, init=Hpsi_guess)
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) atol = 1E-5

  # Test with nsite=1
  Hpsi_guess = random_ttn(t; link_space=4)
  Hpsi = apply(H, psi; alg="fit", nsite=1, nsweeps=4, init=Hpsi_guess)
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) atol = 1E-4
end

@testset "Contract TTN with dangling inds" begin
  nbit = 3
  sites = siteinds("Qubit", nbit)

  # randomMPO does not support linkdims keyword.
  M1 = replaceprime(randomMPO(sites) + randomMPO(sites), 1 => 2, 0 => 1)
  M2 = randomMPO(sites) + randomMPO(sites)
  M12_ref = contract(M1, M2; alg="naive")
  t12_ref = TreeTensorNetwork([M12_ref[v] for v in eachindex(M12_ref)])

  t1 = TreeTensorNetwork([M1[v] for v in eachindex(M1)])
  t2 = TreeTensorNetwork([M2[v] for v in eachindex(M2)])

  # Test with good initial guess
  @test contract(t1, t2; alg="fit", init=t12_ref) ≈ t12_ref rtol = 1e-7
end

nothing
