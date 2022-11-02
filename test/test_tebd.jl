using ITensors
using ITensorNetworks
using Test

ITensors.disable_warn_order()

@testset "Ising TEBD" begin
  dims = (4, 4)
  n = prod(dims)
  g = named_grid(dims)

  h = 0.1

  s = siteinds("S=1/2", g)

  #
  # DMRG comparison
  #
  g_dmrg = rename_vertices(g, cartesian_to_linear(dims))
  ℋ_dmrg = ising(g_dmrg; h)
  s_dmrg = [only(s[v]) for v in vertices(s)]
  H_dmrg = MPO(ℋ_dmrg, s_dmrg)
  ψ_dmrg_init = MPS(s_dmrg, j -> "↑")
  E_dmrg, ψ_dmrg = dmrg(
    H_dmrg, ψ_dmrg_init; nsweeps=20, maxdim=[fill(10, 10); 20], cutoff=1e-8, outputlevel=0
  )

  #
  # PEPS TEBD optimization
  #
  ℋ = ising(g; h)
  χ = 2
  β = 1.0
  Δβ = 0.2

  # Sequence for contracting expectation values
  contract_edges = map(t -> (1, t...), collect(keys(cartesian_to_linear(dims))))
  inner_sequence = reduce((x, y) -> [x, y], contract_edges)

  ψ_init = ITensorNetwork(s, v -> "↑")
  E0 = expect(ℋ, ψ_init; sequence=inner_sequence)
  ψ = tebd(
    group_terms(ℋ, g),
    ψ_init;
    β,
    Δβ,
    cutoff=1e-8,
    maxdim=χ,
    ortho=false,
    print_frequency=typemax(Int),
  )
  E1 = expect(ℋ, ψ; sequence=inner_sequence)
  ψ = tebd(
    group_terms(ℋ, g),
    ψ_init;
    β,
    Δβ,
    cutoff=1e-8,
    maxdim=χ,
    ortho=true,
    print_frequency=typemax(Int),
  )
  E2 = expect(ℋ, ψ; sequence=inner_sequence)

  @test E2 < E1 < E0
  @test E2 ≈ E_dmrg rtol = 1e-5
end
