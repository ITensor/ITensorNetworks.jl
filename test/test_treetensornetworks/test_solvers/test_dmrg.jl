using ITensors
using ITensorNetworks
using Dictionaries
using Random
using Test
using Observers

@testset "MPS DMRG" for nsite in [1, 2]
  N = 10
  cutoff = 1e-12

  s = siteinds("S=1/2", N)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end

  H = mpo(os, s)

  psi = random_mps(s; internal_inds_space=20)

  nsweeps = 10
  maxdim = [10, 20, 40, 100]

  # Compare to `ITensors.MPO` version of `dmrg`
  H_mpo = MPO([H[v] for v in 1:nv(H)])
  psi_mps = MPS([psi[v] for v in 1:nv(psi)])
  e2, psi2 = dmrg(H_mpo, psi_mps; nsweeps, maxdim, outputlevel=0)

  psi = dmrg(H, psi; nsweeps, maxdim, cutoff, nsite, solver_krylovdim=3, solver_maxiter=1)
  @test inner(psi', H, psi) ≈ inner(psi2', H_mpo, psi2)

  # Alias for `ITensorNetworks.dmrg`
  psi = eigsolve(
    H, psi; nsweeps, maxdim, cutoff, nsite, solver_krylovdim=3, solver_maxiter=1
  )
  @test inner(psi', H, psi) ≈ inner(psi2', H_mpo, psi2)

  # Test custom sweep regions
  orig_E = inner(psi', H, psi)
  sweep_regions = [[1], [2], [3], [3], [2], [1]]
  psi = dmrg(H, psi; nsweeps, maxdim, cutoff, sweep_regions)
  new_E = inner(psi', H, psi)
  @test new_E ≈ orig_E
end

@testset "Observers" begin
  N = 10
  cutoff = 1e-12
  s = siteinds("S=1/2", N)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = mpo(os, s)
  psi = random_mps(s; internal_inds_space=20)

  nsweeps = 4
  maxdim = [20, 40, 80, 80]
  cutoff = [1e-10]

  #
  # Make observers
  #
  sweep(; sweep, kw...) = sweep
  sweep_observer! = observer(sweep)

  region(; region, kw...) = region
  energy(; energies, kw...) = energies[1]
  step_observer! = observer(region, sweep, energy)

  psi = dmrg(H, psi; nsweeps, maxdim, cutoff, sweep_observer!, step_observer!)

  #
  # Test out certain values
  #
  @test step_observer![9, :region] == [2, 1]
  @test step_observer![30, :energy] < -4.25
end

@testset "Regression test: Arrays of Parameters" begin
  N = 10
  cutoff = 1e-12

  s = siteinds("S=1/2", N)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end

  H = mpo(os, s)

  psi = random_mps(s; internal_inds_space=20)

  # Choose nsweeps to be less than length of arrays
  nsweeps = 5
  maxdim = [200, 250, 400, 600, 800, 1200, 2000, 2400, 2600, 3000]
  cutoff = [1e-10, 1e-10, 1e-12, 1e-12, 1e-12, 1e-12, 1e-14, 1e-14, 1e-14, 1e-14]

  psi = dmrg(H, psi; nsweeps, maxdim, cutoff)
end

@testset "Tree DMRG" for nsite in [1, 2]
  cutoff = 1e-12

  tooth_lengths = fill(2, 3)
  c = named_comb_tree(tooth_lengths)
  s = siteinds("S=1/2", c)

  os = ITensorNetworks.heisenberg(c)

  H = TTN(os, s)

  psi = random_ttn(s; link_space=20)

  nsweeps = 10
  maxdim = [10, 20, 40, 100]
  sweeps = Sweeps(nsweeps) # number of sweeps is 5
  maxdim!(sweeps, 10, 20, 40, 100) # gradually increase states kept
  cutoff!(sweeps, cutoff)
  psi = dmrg(H, psi; nsweeps, maxdim, cutoff, nsite, solver_krylovdim=3, solver_maxiter=1)

  # Compare to `ITensors.MPO` version of `dmrg`
  linear_order = [4, 1, 2, 5, 3, 6]
  vmap = Dictionary(vertices(s)[linear_order], 1:length(linear_order))
  sline = only.(collect(vertex_data(s)))[linear_order]
  Hline = MPO(relabel_sites(os, vmap), sline)
  psiline = randomMPS(sline; linkdims=20)
  e2, psi2 = dmrg(Hline, psiline, sweeps; outputlevel=0)

  @test inner(psi', H, psi) ≈ inner(psi2', Hline, psi2) atol = 1e-5
end

@testset "Regression test: tree truncation" begin
  maxdim = 4
  nsite = 2
  nsweeps = 10

  c = named_comb_tree((3, 2))
  s = siteinds("S=1/2", c)
  os = ITensorNetworks.heisenberg(c)
  H = TTN(os, s)
  psi = random_ttn(s; link_space=5)
  psi = dmrg(H, psi; nsweeps, maxdim, nsite)

  @test all(edge_data(linkdims(psi)) .<= maxdim)
end

nothing
