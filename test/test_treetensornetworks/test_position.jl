@eval module $(gensym())
using Graphs: vertices
using ITensors: ITensors
using ITensorNetworks: ITensorNetworks, ProjTTN, ttn, environments, position, siteinds
using NamedGraphs: named_comb_tree
using Test

@testset "ProjTTN position" begin
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

  H = ttn(os, s)

  d = Dict()
  for (i, v) in enumerate(vertices(s))
    d[v] = isodd(i) ? "Up" : "Dn"
  end
  states = v -> d[v]
  psi = ttn(s, states)

  # actual test, verifies that position is out of place
  vs = vertices(s)
  PH = ProjTTN(H)
  PH = position(PH, psi, [vs[2]])
  original_keys = deepcopy(keys(environments(PH)))
  # test out-of-placeness of position
  PHc = position(PH, psi, [vs[2], vs[5]])
  @test keys(environments(PH)) == original_keys
  @test keys(environments(PHc)) != original_keys

  if !auto_fermion_enabled
    ITensors.disable_auto_fermion()
  end
end
end
