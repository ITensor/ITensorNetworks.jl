@eval module $(gensym())
using Graphs: vertices
using ITensors.ITensorMPS: MPS
using ITensorNetworks: ttn, expect, random_mps, siteinds
using NamedGraphs: named_comb_tree
using Test: @test, @testset

@testset "MPS expect comparison with ITensors" begin
  N = 25
  s = siteinds("S=1/2", N)
  a = random_mps(s; internal_inds_space=100)
  b = MPS([a[v] for v in vertices(a)])
  res_a = expect("Sz", a)
  res_b = expect(b, "Sz")
  res_a = [res_a[v] for v in vertices(a)]
  @test res_a ≈ res_b rtol = 1e-6
end

@testset "TTN expect" begin
  tooth_lengths = fill(5, 6)
  c = named_comb_tree(tooth_lengths)
  s = siteinds("S=1/2", c)
  d = Dict()
  magnetization = Dict()
  for (i, v) in enumerate(vertices(s))
    d[v] = isodd(i) ? "Up" : "Dn"
    magnetization[v] = isodd(i) ? 0.5 : -0.5
  end
  states = v -> d[v]
  state = ttn(s, states)
  res = expect("Sz", state)
  @test all([isapprox(res[v], magnetization[v]; atol=1e-8) for v in vertices(s)])
end
end
