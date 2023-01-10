using ITensors
using ITensorNetworks
using Random
using Test

@testset "DMRG-X" begin
  function heisenberg(n; h=zeros(n))
    os = OpSum()
    for j in 1:(n - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end
    for j in 1:n
      if h[j] ≠ 0
        os -= h[j], "Sz", j
      end
    end
    return os
  end

  n = 10
  s = siteinds("S=1/2", n)

  Random.seed!(12)

  W = 12
  # Random fields h ∈ [-W, W]
  h = W * (2 * rand(n) .- 1)
  H = MPO(heisenberg(n; h), s)

  initstate = rand(["↑", "↓"], n)
  ψ = MPS(s, initstate)

  dmrg_x_kwargs = (
    nsweeps=20, reverse_step=false, normalize=true, maxdim=20, cutoff=1e-10, outputlevel=0
  )

  ϕ = dmrg_x(ProjMPO(H), ψ; nsite=2, dmrg_x_kwargs...)

  @test inner(ψ', H, ψ) / inner(ψ, ψ) ≈ inner(ϕ', H, ϕ) / inner(ϕ, ϕ) rtol = 1e-1
  @test inner(H, ψ, H, ψ) ≉ inner(ψ', H, ψ)^2 rtol = 1e-7
  @test inner(H, ϕ, H, ϕ) ≈ inner(ϕ', H, ϕ)^2 rtol = 1e-7

  ϕ̃ = dmrg_x(ProjMPO(H), ϕ; nsite=1, dmrg_x_kwargs...)

  @test inner(ψ', H, ψ) / inner(ψ, ψ) ≈ inner(ϕ̃', H, ϕ̃) / inner(ϕ̃, ϕ̃) rtol = 1e-1
  @test inner(H, ϕ̃, H, ϕ̃) ≈ inner(ϕ̃', H, ϕ̃)^2 rtol = 1e-5
  # Sometimes broken, sometimes not
  # @test abs(loginner(ϕ̃, ϕ) / n) ≈ 0.0 atol = 1e-6
end

nothing
