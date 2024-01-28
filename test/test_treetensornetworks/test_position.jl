using ITensors
using ITensorNetworks
using ITensorNetworks: position, environments
using Test

@testset "ProjTTN position copy-safe" begin
  # make a nontrivial TTN state and TTN operator

  auto_fermion_enabled = ITensors.using_auto_fermion()
  use_qns = true
  cutoff = 1e-12

  tooth_lengths = fill(2, 3)
  c = named_comb_tree(tooth_lengths)
  if use_qns  # test whether autofermion breaks things when using non-fermionic QNs
    ITensors.enable_auto_fermion()
  else        # when using no QNs, autofermion breaks # ToDo reference Issue in ITensors
    ITensors.disable_auto_fermion()
  end
  s = siteinds("S=1/2", c; conserve_qns=use_qns)

  os = ITensorNetworks.heisenberg(c)

  H = TTN(os, s)

  d = Dict()
  for (i, v) in enumerate(vertices(s))
    d[v] = isodd(i) ? "Up" : "Dn"
  end
  states = v -> d[v]
  psi = TTN(s, states)

  # actual test, verifies that position is copy safe
  # ToDo: wrap tests so that a failing test does not influence the correctness of the subsequent ones
  vs = vertices(s)
  PH = ProjTTN(H)
  PH = position(PH, psi, [vs[2]])
  original_keys = deepcopy(keys(environments(PH)))
  # test copy-safety of position
  PHc = copy(PH)
  PHc = position(PHc, psi, [vs[2], vs[5]])
  @test keys(environments(PH)) == original_keys
  @test keys(environments(PHc)) != original_keys
  # test copy-safety of position!
  PHc = copy(PH)
  PHc = ITensorNetworks.position!(PHc, psi, [vs[2], vs[5]])
  @test keys(environments(PH)) == original_keys
  @test keys(environments(PHc)) != original_keys
  # test out-of-placeness of position
  PHc = position(PH, psi, [vs[2], vs[5]])
  @test keys(environments(PH)) == original_keys
  @test keys(environments(PHc)) != original_keys
  # test in-placeness of position!
  PHc = copy(PH)
  ITensorNetworks.position!(PHc, psi, [vs[2], vs[5]])
  @test keys(environments(PHc)) != original_keys
  # test that position is copy, regardless of behaviour of Dictionaries (issue #98 in Dictionaries.jl)
  PHc = ITensorNetworks.unsafe_copy(PH)
  PHc = position(PHc, psi, [vs[2], vs[5]])
  @test keys(environments(PH)) == original_keys
  @test keys(environments(PHc)) != original_keys
  # test that position! is itself not copysafe
  # but that copy-safety is due to use of work around in implementation of copy
  PHc = ITensorNetworks.unsafe_copy(PH)
  ITensorNetworks.position!(PHc, psi, [vs[2], vs[5]])
  @test_broken keys(environments(PH)) == original_keys  # make this a proper test once andyferris/Dictionaries.jl#98 is resolved
  @test keys(environments(PHc)) != original_keys

  if !auto_fermion_enabled
    ITensors.disable_auto_fermion()
  end
end
nothing
