using ITensors
using ITensorNetworks
using ITensorNetworks: position
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
  vs = vertices(s)
  PH0 = ProjTTN(H)
  PH0 = position(PH0, psi, [vs[2]])
  PH = copy(PH0)
  PH = position(PH, psi, [vs[2], vs[5]])
  @test keys(PH.environments) != keys(PH0.environments)
  if !auto_fermion_enabled
    ITensors.disable_auto_fermion()
  end
end
