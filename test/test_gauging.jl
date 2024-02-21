using ITensors
using ITensorNetworks
using ITensorNetworks:
  contract_inner,
  symmetric_gauge,
  symmetric_to_vidal_gauge,
  vidal_itn_canonicalness,
  vidal_gauge,
  symmetric_itn_canonicalness,
  update,
  messages,
  BeliefPropagationCache
using NamedGraphs
using Test
using Compat
using Random
using SplitApplyCombine

@testset "gauging" begin
  n = 3
  dims = (n, n)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  χ = 6

  Random.seed!(5467)
  ψ = randomITensorNetwork(s; link_space=χ)
  ψ_symm, bpc = symmetric_gauge(ψ)

  @test symmetric_itn_canonicalness(ψ_symm, bpc) < 1e-5

  #Test we just did a gauge transform and didn't change the overall network
  @test contract_inner(ψ_symm, ψ) /
        sqrt(contract_inner(ψ_symm, ψ_symm) * contract_inner(ψ, ψ)) ≈ 1.0

  ψψ_symm_V2 = ψ_symm ⊗ prime(dag(ψ_symm); sites=[])
  bpc_V2 = BeliefPropagationCache(ψψ_symm_V2, group(v -> v[1], vertices(ψψ_symm_V2)))
  bpc_V2 = update(bpc_V2; maxiters=50)

  for m_e in values(messages(bpc_V2))
    #Test all message tensors are approximately diagonal
    @test diagITensor(vector(diag(only(m_e))), inds(only(m_e))) ≈ only(m_e) atol = 1e-8
  end

  Γ, Λ = vidal_gauge(ψ)
  @test vidal_itn_canonicalness(Γ, Λ) < 1e-5

  Γ, Λ = symmetric_to_vidal_gauge(ψ_symm, bpc)
  @test vidal_itn_canonicalness(Γ, Λ) < 1e-5
end
