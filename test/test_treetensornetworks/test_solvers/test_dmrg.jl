@eval module $(gensym())
using Dictionaries: Dictionary
using Graphs: nv, vertices
using ITensors: ITensors
using ITensors.ITensorMPS: MPO, MPS
using ITensorNetworks:
  ITensorNetworks, OpSum, TTN, apply, dmrg, inner, mpo, random_mps, siteinds
using KrylovKit: eigsolve
using NamedGraphs: named_comb_tree
using Observers: observer
using Test: @test, @test_broken, @testset

@testset "MPS DMRG" for nsites in [1, 2]
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

  psi = dmrg(
    H, psi; nsweeps, maxdim, cutoff, nsites, updater_kwargs=(; krylovdim=3, maxiter=1)
  )
  @test inner(psi', H, psi) ≈ inner(psi2', H_mpo, psi2)

  # Alias for `ITensorNetworks.dmrg`
  psi = eigsolve(
    H, psi; nsweeps, maxdim, cutoff, nsites, updater_kwargs=(; krylovdim=3, maxiter=1)
  )
  @test inner(psi', H, psi) ≈ inner(psi2', H_mpo, psi2)

  # Test custom sweep regions #BROKEN, ToDo: Make proper custom sweep regions for test
  #=
  orig_E = inner(psi', H, psi)
  sweep_regions = [[1], [2], [3], [3], [2], [1]]
  psi = dmrg(H, psi; nsweeps, maxdim, cutoff, sweep_regions)
  new_E = inner(psi', H, psi)
  @test new_E ≈ orig_E
  =#
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
  sweep(; which_sweep, kw...) = which_sweep
  sweep_observer! = observer(sweep)

  region(; which_region_update, sweep_plan, kw...) = first(sweep_plan[which_region_update])
  energy(; eigvals, kw...) = eigvals[1]
  region_observer! = observer(region, sweep, energy)

  psi = dmrg(H, psi; nsweeps, maxdim, cutoff, sweep_observer!, region_observer!)

  #
  # Test out certain values
  #
  @test region_observer![9, :region] == [2, 1]
  @test region_observer![30, :energy] < -4.25
end

@testset "Cache to Disk" begin
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
  psi = random_mps(s; internal_inds_space=10)

  nsweeps = 4
  maxdim = [10, 20, 40, 80]

  @test_broken psi = dmrg(
    H,
    psi;
    nsweeps,
    maxdim,
    cutoff,
    outputlevel=2,
    transform_operator=ITensorNetworks.cache_operator_to_disk,
    transform_operator_kwargs=(; write_when_maxdim_exceeds=11),
  )
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

@testset "Tree DMRG" for nsites in [2]
  cutoff = 1e-12

  tooth_lengths = fill(2, 3)
  c = named_comb_tree(tooth_lengths)

  @testset "SVD approach" for use_qns in [false, true]
    auto_fermion_enabled = ITensors.using_auto_fermion()
    if use_qns  # test whether autofermion breaks things when using non-fermionic QNs
      ITensors.enable_auto_fermion()
    else        # when using no QNs, autofermion breaks # ToDo reference Issue in ITensors
      ITensors.disable_auto_fermion()
    end
    s = siteinds("S=1/2", c; conserve_qns=use_qns)

    os = ITensorNetworks.heisenberg(c)

    H = TTN(os, s)

    # make init_state
    d = Dict()
    for (i, v) in enumerate(vertices(s))
      d[v] = isodd(i) ? "Up" : "Dn"
    end
    states = v -> d[v]
    psi = TTN(s, states)

    #    psi = random_ttn(s; link_space=20) #FIXME: random_ttn broken for QN conserving case

    nsweeps = 10
    maxdim = [10, 20, 40, 100]
    @show use_qns
    psi = dmrg(
      H, psi; nsweeps, maxdim, cutoff, nsites, updater_kwargs=(; krylovdim=3, maxiter=1)
    )

    # Compare to `ITensors.MPO` version of `dmrg`
    linear_order = [4, 1, 2, 5, 3, 6]
    vmap = Dictionary(vertices(s)[linear_order], 1:length(linear_order))
    sline = only.(collect(vertex_data(s)))[linear_order]
    Hline = MPO(relabel_sites(os, vmap), sline)
    psiline = randomMPS(sline, i -> isodd(i) ? "Up" : "Dn"; linkdims=20)
    e2, psi2 = dmrg(Hline, psiline; nsweeps, maxdim, cutoff, outputlevel=0)

    @test inner(psi', H, psi) ≈ inner(psi2', Hline, psi2) atol = 1e-5

    if !auto_fermion_enabled
      ITensors.disable_auto_fermion()
    end
  end
end

@testset "Tree DMRG for Fermions" for nsites in [2] #ToDo: change to [1,2] when random_ttn works with QNs
  auto_fermion_enabled = ITensors.using_auto_fermion()
  use_qns = true
  cutoff = 1e-12
  nsweeps = 10
  maxdim = [10, 20, 40, 100]

  # setup model
  tooth_lengths = fill(2, 3)
  c = named_comb_tree(tooth_lengths)
  s = siteinds("Electron", c; conserve_qns=use_qns)
  U = 2.0
  t = 1.3
  tp = 0.6
  os = ITensorNetworks.hubbard(c; U, t, tp)

  # for conversion to ITensors.MPO
  linear_order = [4, 1, 2, 5, 3, 6]
  vmap = Dictionary(vertices(s)[linear_order], 1:length(linear_order))
  sline = only.(collect(vertex_data(s)))[linear_order]

  # get MPS / MPO with JW string result
  ITensors.disable_auto_fermion()
  Hline = MPO(relabel_sites(os, vmap), sline)
  psiline = randomMPS(sline, i -> isodd(i) ? "Up" : "Dn"; linkdims=20)
  e_jw, psi_jw = dmrg(Hline, psiline; nsweeps, maxdim, cutoff, outputlevel=0)
  ITensors.enable_auto_fermion()

  # now get auto-fermion results 
  H = TTN(os, s)
  # make init_state
  d = Dict()
  for (i, v) in enumerate(vertices(s))
    d[v] = isodd(i) ? "Up" : "Dn"
  end
  states = v -> d[v]
  psi = TTN(s, states)
  psi = dmrg(
    H, psi; nsweeps, maxdim, cutoff, nsites, updater_kwargs=(; krylovdim=3, maxiter=1)
  )

  # Compare to `ITensors.MPO` version of `dmrg`
  Hline = MPO(relabel_sites(os, vmap), sline)
  psiline = randomMPS(sline, i -> isodd(i) ? "Up" : "Dn"; linkdims=20)
  e2, psi2 = dmrg(Hline, psiline; nsweeps, maxdim, cutoff, outputlevel=0)

  @test inner(psi', H, psi) ≈ inner(psi2', Hline, psi2) atol = 1e-5
  @test e2 ≈ e_jw atol = 1e-5
  @test inner(psi2', Hline, psi2) ≈ e_jw atol = 1e-5

  if !auto_fermion_enabled
    ITensors.disable_auto_fermion()
  end
end

@testset "Regression test: tree truncation" begin
  maxdim = 4
  nsites = 2
  nsweeps = 10

  c = named_comb_tree((3, 2))
  s = siteinds("S=1/2", c)
  os = ITensorNetworks.heisenberg(c)
  H = TTN(os, s)
  psi = random_ttn(s; link_space=5)
  psi = dmrg(H, psi; nsweeps, maxdim, nsites)

  @test all(edge_data(linkdims(psi)) .<= maxdim)
end
end
