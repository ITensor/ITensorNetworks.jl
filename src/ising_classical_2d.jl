module Models

  using ITensors

  struct Model{model} end
  Model(s::AbstractString) = Model{Symbol(s)}()
  macro Model_str(s)
    :(Model{$(Expr(:quote, Symbol(s)))})
  end

  #mpo_itensor(m::String, args...; kwargs...) =
  #  mpo_itensor(Model(m), args...; kwargs...)

  local_boltzmann_weight(m::String, args...; kwargs...) =
    local_boltzmann_weight(Model(m), args...; kwargs...)

  f(λ₊, λ₋) = [
    (λ₊ + λ₋)/2 (λ₊ - λ₋)/2
    (λ₊ - λ₋)/2 (λ₊ + λ₋)/2
  ]

  function sqrt_bond_matrix(; β::Real, J::Real=1.0)
    # Alternative method
    #Q = [exp(β * J) exp(-β * J); exp(-β * J) exp(β * J)]
    #return √Q
    λ₊ = √(exp(β * J) + exp(-β * J))
    λ₋ = √(exp(β * J) - exp(-β * J))
    return f(λ₊, λ₋)
  end

  # The local Boltzmann weight for the Ising
  # model in 1 dimension
  function local_boltzmann_weight(
    ::Model"ising",
    ::Val{1};
    β::Real,
    J::Real=1.0,
    sz::Bool=false,
  )
    d = 2 # local dimension of the Ising local Boltzmann factor
    s, s′ = Index.((d, d))
    T = ITensor(s, s′)
    for i in 1:d
      T[i, i] = 1.0
    end
    if sz
      T[1, 1] = -T[1, 1]
    end
    s̃, s̃′ = sim.((s, s′))
    T̃ = T * δ(s, s̃) * δ(s′, s̃′)
    sqrtQ = sqrt_bond_matrix(; β=β, J=J)
    @show sqrtQ
    X = itensor(vec(sqrtQ), s̃, s)
    X′ = itensor(vec(sqrtQ), s̃′, s′)
    return array(permute(T̃ * X′ * X, s, s′))
  end

  # The local Boltzmann weight for the Ising
  # model in 2 dimensions
  function local_boltzmann_weight(
    ::Model"ising",
    ::Val{2};
    β::Real,
    J::Real=1.0,
    sz::Bool=false,
  )
    d = 2 # local dimension of the Ising local Boltzmann factor
    sₕ, sₕ′ = Index.((d, d))
    sᵥ, sᵥ′ = Index.((d, d))
    @assert dim(sₕ) == dim(sᵥ)
    d = dim(sₕ)
    T = ITensor(sₕ, sₕ′, sᵥ, sᵥ′)
    for i in 1:d
      T[i, i, i, i] = 1.0
    end
    if sz
      T[1, 1, 1, 1] = -T[1, 1, 1, 1]
    end
    s̃ₕ, s̃ₕ′, s̃ᵥ, s̃ᵥ′ = sim.((sₕ, sₕ′, sᵥ, sᵥ′))
    T̃ = T * δ(sₕ, s̃ₕ) * δ(sₕ′, s̃ₕ′) * δ(sᵥ, s̃ᵥ) * δ(sᵥ′, s̃ᵥ′)
    X = sqrt_bond_matrix(; β, J)
    Xₕ = itensor(vec(X), s̃ₕ, sₕ)
    Xₕ′ = itensor(vec(X), s̃ₕ′, sₕ′)
    Xᵥ = itensor(vec(X), s̃ᵥ, sᵥ)
    Xᵥ′ = itensor(vec(X), s̃ᵥ′, sᵥ′)
    return array(permute(T̃ * Xₕ′ * Xᵥ′ * Xₕ * Xᵥ, sₕ, sₕ′, sᵥ, sᵥ′))
  end

  function mpo_itensor(
    m::Model,
    pair_sₕ::Pair{<:Index,<:Index},
    pair_sᵥ::Pair{<:Index,<:Index};
    kwargs...
  )
    sₕ, sₕ′ = pair_sₕ
    sᵥ, sᵥ′ = pair_sᵥ
    return itensor(mpo_array(m; kwargs...), sₕ, sₕ′, sᵥ, sᵥ′)
  end

  ## function ising_mpo(sₕ::Index, sᵥ::Index, args...; kwargs...)
  ##   return ising_mpo(sₕ => sₕ', sᵥ => sᵥ', args...; kwargs...)
  ## end

  critical_point(::Model"ising") = 0.5 * log(√2 + 1)
  #const βc = critical_point(Model("ising"))

  function free_energy(::Model"ising"; β::Real, J::Real=1.0)
    k = β * J
    c = cosh(2 * k)
    s = sinh(2 * k)
    xmin = 0.0
    xmax = π
    integrand(x) = log(c^2 + √(s^4 + 1 - 2 * s^2 * cos(x)))
    integral, err = quadgk(integrand, xmin, xmax)::Tuple{Float64,Float64}
    return -(log(2) + integral / π) / (2 * β)
  end

  function magnetization(::Model"ising"; β::Real)
    β > βc && return (1 - sinh(2 * β)^(-4))^(1 / 8)
    return 0.0
  end

end

