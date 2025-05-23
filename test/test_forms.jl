@eval module $(gensym())
using DataGraphs: underlying_graph
using Graphs: nv
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using ITensorNetworks:
  BeliefPropagationCache,
  BilinearFormNetwork,
  LinearFormNetwork,
  QuadraticFormNetwork,
  bra_network,
  bra_vertex,
  dual_index_map,
  environment,
  flatten_siteinds,
  inner,
  ket_network,
  ket_vertex,
  operator_network,
  random_tensornetwork,
  scalar,
  siteinds,
  state_vertices,
  tensornetwork,
  union_all_inds,
  update
using ITensors: Index, contract, dag, inds, prime, random_itensor, sim
using LinearAlgebra: norm
using StableRNGs: StableRNG
using TensorOperations: TensorOperations
using Test: @test, @testset
@testset "FormNetworks" begin
  g = named_grid((1, 4))
  s = siteinds("S=1/2", g)
  s_operator = union_all_inds(s, prime(s))
  χ, D = 2, 3
  rng = StableRNG(1234)
  ψket = random_tensornetwork(rng, s; link_space=χ)
  ψbra = random_tensornetwork(rng, s; link_space=χ)
  A = random_tensornetwork(rng, s_operator; link_space=D)

  lf = LinearFormNetwork(ψbra, ψket)
  @test nv(lf) == nv(ψket) + nv(ψbra)
  @test isempty(flatten_siteinds(lf))

  blf = BilinearFormNetwork(A, ψbra, ψket)
  @test nv(blf) == nv(ψket) + nv(ψbra) + nv(A)
  @test isempty(flatten_siteinds(blf))

  @test underlying_graph(ket_network(blf)) == underlying_graph(ψket)
  @test underlying_graph(operator_network(blf)) == underlying_graph(A)
  @test underlying_graph(bra_network(blf)) == underlying_graph(ψbra)

  lf = LinearFormNetwork(blf)
  @test underlying_graph(ket_network(lf)) == underlying_graph(ψket)

  qf = QuadraticFormNetwork(ψket)
  @test nv(qf) == 3 * nv(ψket)
  @test isempty(flatten_siteinds(qf))

  qf = QuadraticFormNetwork(A, ψket)
  @test nv(qf) == 2 * nv(ψket) + nv(A)
  @test isempty(flatten_siteinds(qf))

  v = (1, 1)
  rng = StableRNG(1234)
  new_tensor = random_itensor(rng, inds(ψket[v]))
  qf_updated = update(qf, v, copy(new_tensor))

  @test tensornetwork(qf_updated)[bra_vertex(qf_updated, v)] ≈
    dual_index_map(qf_updated)(dag(new_tensor))
  @test tensornetwork(qf_updated)[ket_vertex(qf_updated, v)] ≈ new_tensor

  @test underlying_graph(ket_network(qf)) == underlying_graph(ψket)
  @test underlying_graph(operator_network(qf)) == underlying_graph(A)

  ∂qf_∂v = only(environment(qf, state_vertices(qf, [v]); alg="exact"))
  @test (∂qf_∂v) * (qf[ket_vertex(qf, v)] * qf[bra_vertex(qf, v)]) ≈ contract(qf)

  ∂qf_∂v_bp = environment(qf, state_vertices(qf, [v]); alg="bp", update_cache=false)
  ∂qf_∂v_bp = contract(∂qf_∂v_bp)
  ∂qf_∂v_bp /= norm(∂qf_∂v_bp)
  ∂qf_∂v /= norm(∂qf_∂v)
  @test ∂qf_∂v_bp != ∂qf_∂v

  ∂qf_∂v_bp = environment(qf, state_vertices(qf, [v]); alg="bp", update_cache=true)
  ∂qf_∂v_bp = contract(∂qf_∂v_bp)
  ∂qf_∂v_bp /= norm(∂qf_∂v_bp)
  @test ∂qf_∂v_bp ≈ ∂qf_∂v

  #Test having non-uniform number of site indices per vertex
  g = named_comb_tree((3, 3))
  s = siteinds("S=1/2", g)
  s = union_all_inds(s, sim(s))
  s[(1, 1)] = Index[]
  s[(3, 3)] = Index[first(s[(3, 3)])]
  χ = 2
  rng = StableRNG(1234)
  ψket = random_tensornetwork(rng, ComplexF64, s; link_space=χ)
  ψbra = random_tensornetwork(rng, ComplexF64, s; link_space=χ)

  blf = BilinearFormNetwork(ψbra, ψket)
  @test scalar(blf; alg="exact") ≈ inner(ψbra, ψket; alg="exact")

  lf = LinearFormNetwork(ψbra, ψket)
  @test scalar(lf; alg="exact") ≈ inner(ψbra, ψket; alg="exact")

  qf = QuadraticFormNetwork(ψket)
  @test scalar(qf; alg="exact") ≈ inner(ψket, ψket; alg="exact")
end
end
