using ITensors
using ITensors: contract
using Graphs
using NamedGraphs
using ITensorNetworks
using ITensorNetworks:
  delta_network,
  update,
  tensornetwork,
  bra_vertex_map,
  ket_vertex_map,
  dual_index_map,
  bra_network,
  ket_network,
  operator_network,
  contract_with_BP,
  group
using Test
using Random

@testset "FormNetworks" begin
  g = named_grid((1, 4))
  s_ket = siteinds("S=1/2", g)
  s_bra = prime(s_ket; links=[])
  s_operator = union_all_inds(s_bra, s_ket)
  χ, D = 2, 3
  Random.seed!(1234)
  ψket = randomITensorNetwork(s_ket; link_space=χ)
  ψbra = randomITensorNetwork(s_bra; link_space=χ)
  A = randomITensorNetwork(s_operator; link_space=D)

  blf = BilinearFormNetwork(A, ψbra, ψket)
  @test nv(blf) == nv(ψket) + nv(ψbra) + nv(A)
  @test isempty(externalinds(blf))

  @test underlying_graph(ket_network(blf)) == underlying_graph(ψket)
  @test underlying_graph(operator_network(blf)) == underlying_graph(A)
  @test underlying_graph(bra_network(blf)) == underlying_graph(ψbra)

  qf = QuadraticFormNetwork(A, ψket)
  @test nv(qf) == 2 * nv(ψbra) + nv(A)
  @test isempty(externalinds(qf))

  v = (1, 1)
  new_tensor = randomITensor(inds(ψket[v]))
  qf_updated = update(qf, v, copy(new_tensor))

  @test tensornetwork(qf_updated)[bra_vertex_map(qf_updated)(v)] ≈
    dual_index_map(qf_updated)(dag(new_tensor))
  @test tensornetwork(qf_updated)[ket_vertex_map(qf_updated)(v)] ≈ new_tensor

  @test underlying_graph(ket_network(qf)) == underlying_graph(ψket)
  @test underlying_graph(operator_network(qf)) == underlying_graph(A)
end

@testset "FormNetworks for TTN" begin
  Random.seed!(1234)
  Lx, Ly = 3, 3
  χ = 2
  g = named_comb_tree((Lx, Ly))
  s = siteinds("S=1/2", g)
  y = TTN(randomITensorNetwork(ComplexF64, s; link_space=χ))
  x = TTN(randomITensorNetwork(ComplexF64, s; link_space=χ))

  A = TTN(ITensorNetworks.heisenberg(s), s)
  #First lets do it with the flattened version of the network
  xy = inner_network(x, y; combine_linkinds=true)
  xy_scalar = contract(xy)[]
  xy_scalar_bp = contract_with_BP(xy)

  @test_broken xy_scalar ≈ xy_scalar_bp

  #Now lets keep it unflattened and do Block BP to keep the partitioned graph as a tree
  xy = inner_network(x, y; combine_linkinds=false)
  xy_scalar = contract(xy)[]
  xy_scalar_bp = contract_with_BP(xy; partitioning=group(v -> first(v), vertices(xy)))

  @test xy_scalar ≈ xy_scalar_bp
  # test contraction of three layers for expectation values
  # for TTN inner with this signature passes via contract_with_BP
  @test inner(prime(x), A, y) ≈
    inner(x, apply(A, y; nsweeps=10, maxdim=16, cutoff=1e-10, init=y)) rtol = 1e-6
end
nothing
