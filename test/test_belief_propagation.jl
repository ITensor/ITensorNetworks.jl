using ITensorNetworks
using ITensorNetworks:
  construct_initial_mts,
  get_single_site_expec,
  update_all_mts,
  ising_network,
  get_two_site_expec,
  two_site_rdm_bp
using Test
using Compat
using ITensors
#ASSUME ONE CAN INSTALL THIS, MIGHT FAIL FOR WINDOWS
using Metis
using LinearAlgebra

@testset "belief_propagation" begin

  #FIRST TEST SINGLE SITE ON AN MPS, SHOULD BE ALMOST EXACT
  dims = (1, 6)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  χ = 4
  ψ = randomITensorNetwork(s; link_space=χ)

  ψψ = norm_sqr_network(ψ; flatten=true, map_bra_linkinds=prime)
  combiners = linkinds_combiners(ψψ)
  ψψ = combine_linkinds(ψψ, combiners)

  v = (1, 3)
  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  ψOψ = inner_network(ψ, Oψ; flatten=true, map_bra_linkinds=prime)
  ψOψ = combine_linkinds(ψOψ, combiners)

  contract_seq = contraction_sequence(ψψ)
  actual_sz =
    ITensors.contract(ψOψ; sequence=contract_seq)[] /
    ITensors.contract(ψψ; sequence=contract_seq)[]

  niters, nsites = 20, 1
  mts = construct_initial_mts(ψψ, nsites; init=(I...) -> @compat allequal(I) ? 1 : 0)
  mts = update_all_mts(ψψ, mts, niters)
  bp_sz = get_single_site_expec(ψψ, mts, ψOψ, v)

  @test abs.(bp_sz - actual_sz) <= 0.00001

  #NOW TEST TWO_SITE_EXPEC TAKING ON THE PARTITION FUNCTION OF THE RECTANGULAR ISING. SHOULD BE REASONABLE AND 
  #INDEPENDENT OF INIT CONDITIONS, FOR SMALL BETA
  dims = (3, 4)
  g = named_grid(dims)
  s = IndsNetwork(g; link_space=2)
  beta = 0.2
  v1, v2 = (2, 3), (3, 3)
  ψψ = ising_network(s, beta)
  ψOψ = ising_network(s, beta; szverts=[v1, v2])

  contract_seq = contraction_sequence(ψψ)
  actual_szsz =
    ITensors.contract(ψOψ; sequence=contract_seq)[] /
    ITensors.contract(ψψ; sequence=contract_seq)[]

  niters, nsites = 10, 2
  mts = construct_initial_mts(ψψ, nsites; init=(I...) -> @compat allequal(I) ? 1 : 0)
  mts = update_all_mts(ψψ, mts, niters)
  bp_szsz = get_two_site_expec(ψψ, mts, ψOψ, v1, v2)

  @test abs.(bp_szsz - actual_szsz) <= 0.05

  #FINALLY, TEST FORMING OF A TWO SITE RDM. JUST CHECK THAT IS HAS THE CORRECT SIZE, TRACE AND IS PSD
  dims = (4, 4)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  v1, v2 = (2, 2), (2, 3)
  χ = 3
  ψ = randomITensorNetwork(s; link_space=χ)
  ψψ = norm_sqr_network(ψ; flatten=true, map_bra_linkinds=prime)
  combiners = linkinds_combiners(ψψ)
  ψψ = combine_linkinds(ψψ, combiners)

  niters, nsites = 20, 2
  mts = construct_initial_mts(ψψ, nsites; init=(I...) -> @compat allequal(I) ? 1 : 0)
  mts = update_all_mts(ψψ, mts, niters)
  approx_rdm = two_site_rdm_bp(ψψ, ψ, mts, v1, v2, s, combiners)

  @test abs(tr(approx_rdm) - 1.0) < 0.00000001
  eigs = eigvals(approx_rdm)
  @test all(>=(0), real(eigs)) && all(==(0), imag(eigs))
  @test size(approx_rdm) == (4, 4)
end
