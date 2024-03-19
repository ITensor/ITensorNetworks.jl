using ITensors
using ITensors: contract
using ITensorNetworks
using ITensorNetworks: contract_with_BP, group
using Test

@testset "TTN constructor defaulting to link_space=1" begin
  tooth_lengths = fill(5, 6)
  c = named_comb_tree(tooth_lengths)
  s = siteinds("S=1/2", c)
  d = Dict()
  for (i, v) in enumerate(vertices(s))
    d[v] = isodd(i) ? "Up" : "Dn"
  end
  states = v -> d[v]
  #test a few signatures
  state = TTN(s, states)
  lds = edge_data(linkdims(state))
  @test all([isone(lds[k]) for k in keys(lds)])
  state = TTN(s)
  lds = edge_data(linkdims(state))
  @test all([isone(lds[k]) for k in keys(lds)])
end

@testset "Inner products for TTN via BP" begin
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
