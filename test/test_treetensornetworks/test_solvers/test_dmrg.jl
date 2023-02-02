using ITensors
using ITensorNetworks
using Dictionaries
using Random
using Test

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
  e2, psi2 = dmrg(H_mpo, psi_mps; nsweeps, maxdim, normalize=false, outputlevel=0)

  ## sweeps = Sweeps(nsweeps) # number of sweeps is 5
  ## maxdim!(sweeps, 10, 20, 40, 100) # gradually increase states kept
  ## cutoff!(sweeps, cutoff)

  psi = dmrg(H, psi; nsweeps, maxdim, cutoff, nsite, solver_krylovdim=3, solver_maxiter=1)
  @test inner(psi', H, psi) ≈ inner(psi2', H_mpo, psi2)

  # Alias for `ITensorNetworks.dmrg`
  psi = eigsolve(
    H, psi; nsweeps, maxdim, cutoff, nsite, solver_krylovdim=3, solver_maxiter=1
  )
  @test inner(psi', H, psi) ≈ inner(psi2', H_mpo, psi2)
end

@testset "Tree DMRG" for nsite in [1, 2]
  cutoff = 1e-12

  tooth_lengths = fill(2, 3)
  root_vertex = (3, 2)
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
  e2, psi2 = dmrg(Hline, psiline, sweeps; normalize=false, outputlevel=0)

  @test inner(psi', H, psi) ≈ inner(psi2', Hline, psi2) atol = 1e-5
end

nothing
