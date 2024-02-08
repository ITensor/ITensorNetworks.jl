using ITensors
using Graphs
using NamedGraphs
using ITensorNetworks
using ITensorNetworks: delta_network, bra_ket_vertices, gradient
using Test
using Random

@testset "Forms" begin
  g = named_grid((1, 4))
  s = siteinds("S=1/2", g)
  s_operator = union_all_inds(s, prime(s; links=[]))
  χ = 2
  Random.seed!(1234)
  ψket = randomITensorNetwork(s; link_space=χ)
  ψbra = randomITensorNetwork(s; link_space=χ)
  A = delta_network(s_operator)

  blf = BilinearForm(ψbra, A, ψket)
  @test nv(blf) == nv(ψket) + nv(ψbra) + nv(A)
  @test isempty(externalinds(blf))

  qf = QuadraticForm(ψbra, A)
  @test nv(qf) == nv(ψket) + nv(ψbra) + nv(A)
  @test isempty(externalinds(qf))
  v = (1, 1)

  qf_gradient_v = gradient(qf, [v]; alg="Exact")
  z_qf = ITensors.contract([qf_gradient_v, dag(prime(ψbra[v])), ψbra[v]])[]
  @test z_qf ≈ ITensors.contract(qf)[]
end
