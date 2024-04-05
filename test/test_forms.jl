@eval module $(gensym())
using DataGraphs: underlying_graph
using Graphs: nv
using NamedGraphs
using ITensorNetworks:
  BeliefPropagationCache,
  BilinearFormNetwork,
  QuadraticFormNetwork,
  bra_network,
  bra_vertex,
  delta_network,
  dual_index_map,
  environment,
  externalinds,
  ket_network,
  ket_vertex,
  operator_network,
  random_tensornetwork,
  siteinds,
  tensornetwork,
  union_all_inds,
  update
using ITensors: contract, dag, inds, prime, randomITensor
using LinearAlgebra: norm
using Test: @test, @testset
using Random: Random

@testset "FormNetworks" begin
  g = named_grid((1, 4))
  s = siteinds("S=1/2", g)
  s_operator = union_all_inds(s, prime(s))
  χ, D = 2, 3
  Random.seed!(1234)
  ψket = random_tensornetwork(s_ket; link_space=χ)
  ψbra = random_tensornetwork(s_bra; link_space=χ)
  A = random_tensornetwork(s_operator; link_space=D)

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

  @test tensornetwork(qf_updated)[bra_vertex(qf_updated, v)] ≈
    dual_index_map(qf_updated)(dag(new_tensor))
  @test tensornetwork(qf_updated)[ket_vertex(qf_updated, v)] ≈ new_tensor

  @test underlying_graph(ket_network(qf)) == underlying_graph(ψket)
  @test underlying_graph(operator_network(qf)) == underlying_graph(A)

  ∂qf_∂v = only(environment(qf, [v]))
  @test (∂qf_∂v) * (qf[ket_vertex(qf, v)] * qf[bra_vertex(qf, v)]) ≈ contract(qf)

  ∂qf_∂v_bp = environment(qf, [v]; alg="bp", update_cache=false)
  ∂qf_∂v_bp = contract(∂qf_∂v_bp)
  ∂qf_∂v_bp /= norm(∂qf_∂v_bp)
  ∂qf_∂v /= norm(∂qf_∂v)
  @test ∂qf_∂v_bp != ∂qf_∂v

  ∂qf_∂v_bp = environment(qf, [v]; alg="bp", update_cache=true)
  ∂qf_∂v_bp = contract(∂qf_∂v_bp)
  ∂qf_∂v_bp /= norm(∂qf_∂v_bp)
  @test ∂qf_∂v_bp ≈ ∂qf_∂v
end
end
