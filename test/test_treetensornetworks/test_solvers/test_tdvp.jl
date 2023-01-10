using ITensors
using ITensorNetworks
using KrylovKit: exponentiate
using Observers
using Random
using Test

@testset "MPS TDVP" begin
  @testset "Basic TDVP" begin
    N = 10
    cutoff = 1e-12

    s = siteinds("S=1/2", N)

    os = OpSum()
    for j in 1:(N - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end

    H = MPO(os, s)

    ψ0 = randomMPS(s; linkdims=10)

    # Time evolve forward:
    ψ1 = tdvp(H, -0.1im, ψ0; nsweeps=1, cutoff, nsite=1)

    @test ψ1 ≈ tdvp(-0.1im, H, ψ0; nsweeps=1, cutoff, nsite=1)
    @test ψ1 ≈ tdvp(H, ψ0, -0.1im; nsweeps=1, cutoff, nsite=1)
    #Different backend solvers, default solver_backend = "applyexp"
    @test ψ1 ≈
      tdvp(H, ψ0, -0.1im; nsweeps=1, cutoff, nsite=1, solver_backend="exponentiate")

    @test norm(ψ1) ≈ 1.0

    ## Should lose fidelity:
    #@test abs(inner(ψ0,ψ1)) < 0.9

    # Average energy should be conserved:
    @test real(inner(ψ1', H, ψ1)) ≈ inner(ψ0', H, ψ0)

    # Time evolve backwards:
    ψ2 = tdvp(H, +0.1im, ψ1; nsweeps=1, cutoff)

    #Different backend solvers, default solver_backend = "applyexp"
    @test ψ2 ≈ tdvp(H, +0.1im, ψ1; nsweeps=1, cutoff, solver_backend="exponentiate")

    @test norm(ψ2) ≈ 1.0

    # Should rotate back to original state:
    @test abs(inner(ψ0, ψ2)) > 0.99
  end

  @testset "TDVP: Sum of Hamiltonians" begin
    N = 10
    cutoff = 1e-10

    s = siteinds("S=1/2", N)

    os1 = OpSum()
    for j in 1:(N - 1)
      os1 += 0.5, "S+", j, "S-", j + 1
      os1 += 0.5, "S-", j, "S+", j + 1
    end
    os2 = OpSum()
    for j in 1:(N - 1)
      os2 += "Sz", j, "Sz", j + 1
    end

    H1 = MPO(os1, s)
    H2 = MPO(os2, s)
    Hs = [H1, H2]

    ψ0 = randomMPS(s; linkdims=10)

    ψ1 = tdvp(Hs, -0.1im, ψ0; nsweeps=1, cutoff, nsite=1)

    @test ψ1 ≈ tdvp(-0.1im, Hs, ψ0; nsweeps=1, cutoff, nsite=1)
    @test ψ1 ≈ tdvp(Hs, ψ0, -0.1im; nsweeps=1, cutoff, nsite=1)

    #Different backend solvers, default solver_backend = "applyexp"
    @test ψ1 ≈
      tdvp(Hs, ψ0, -0.1im; nsweeps=1, cutoff, nsite=1, solver_backend="exponentiate")

    @test norm(ψ1) ≈ 1.0

    ## Should lose fidelity:
    #@test abs(inner(ψ0,ψ1)) < 0.9

    # Average energy should be conserved:
    @test real(sum(H -> inner(ψ1', H, ψ1), Hs)) ≈ sum(H -> inner(ψ0', H, ψ0), Hs)

    # Time evolve backwards:
    ψ2 = tdvp(Hs, +0.1im, ψ1; nsweeps=1, cutoff)

    #Different backend solvers, default solver_backend = "applyexp"
    @test ψ2 ≈ tdvp(Hs, +0.1im, ψ1; nsweeps=1, cutoff, solver_backend="exponentiate")

    @test norm(ψ2) ≈ 1.0

    # Should rotate back to original state:
    @test abs(inner(ψ0, ψ2)) > 0.99
  end

  @testset "Custom solver in TDVP" begin
    N = 10
    cutoff = 1e-12

    s = siteinds("S=1/2", N)

    os = OpSum()
    for j in 1:(N - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end

    H = MPO(os, s)

    ψ0 = randomMPS(s; linkdims=10)

    function solver(PH, t, psi0; kwargs...)
      solver_kwargs = (;
        ishermitian=true, tol=1e-12, krylovdim=30, maxiter=100, verbosity=0, eager=true
      )
      psi, info = exponentiate(PH, t, psi0; solver_kwargs...)
      return psi, info
    end

    ψ1 = tdvp(solver, H, -0.1im, ψ0; cutoff, nsite=1)

    @test norm(ψ1) ≈ 1.0

    ## Should lose fidelity:
    #@test abs(inner(ψ0,ψ1)) < 0.9

    # Average energy should be conserved:
    @test real(inner(ψ1', H, ψ1)) ≈ inner(ψ0', H, ψ0)

    # Time evolve backwards:
    ψ2 = tdvp(H, +0.1im, ψ1; cutoff)

    @test norm(ψ2) ≈ 1.0

    # Should rotate back to original state:
    @test abs(inner(ψ0, ψ2)) > 0.99
  end

  @testset "Accuracy Test" begin
    N = 4
    tau = 0.1
    ttotal = 1.0
    cutoff = 1e-12

    s = siteinds("S=1/2", N; conserve_qns=false)

    os = OpSum()
    for j in 1:(N - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end
    H = MPO(os, s)
    HM = prod(H)

    Ut = exp(-im * tau * HM)

    psi = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    psi2 = deepcopy(psi)
    psix = prod(psi)

    Sz_tdvp = Float64[]
    Sz_tdvp2 = Float64[]
    Sz_exact = Float64[]

    c = div(N, 2)
    Szc = op("Sz", s[c])

    Nsteps = Int(ttotal / tau)
    for step in 1:Nsteps
      psix = noprime(Ut * psix)
      psix /= norm(psix)

      psi = tdvp(
        H,
        -im * tau,
        psi;
        cutoff,
        normalize=false,
        solver_tol=1e-12,
        solver_maxiter=500,
        solver_krylovdim=25,
      )
      push!(Sz_tdvp, real(expect(psi, "Sz"; sites=c:c)[1]))

      psi2 = tdvp(
        H,
        -im * tau,
        psi2;
        cutoff,
        normalize=false,
        solver_tol=1e-12,
        solver_maxiter=500,
        solver_krylovdim=25,
        solver_solver_backend="exponentiate",
      )
      push!(Sz_tdvp2, real(expect(psi2, "Sz"; sites=c:c)[1]))

      push!(Sz_exact, real(scalar(dag(prime(psix, s[c])) * Szc * psix)))
      F = abs(scalar(dag(psix) * prod(psi)))
    end

    @test norm(Sz_tdvp - Sz_exact) < 1e-5
    @test norm(Sz_tdvp2 - Sz_exact) < 1e-5
  end

  @testset "TEBD Comparison" begin
    N = 10
    cutoff = 1e-12
    tau = 0.1
    ttotal = 1.0

    s = siteinds("S=1/2", N; conserve_qns=true)

    os = OpSum()
    for j in 1:(N - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end

    H = MPO(os, s)

    gates = ITensor[]
    for j in 1:(N - 1)
      s1 = s[j]
      s2 = s[j + 1]
      hj =
        op("Sz", s1) * op("Sz", s2) +
        1 / 2 * op("S+", s1) * op("S-", s2) +
        1 / 2 * op("S-", s1) * op("S+", s2)
      Gj = exp(-1.0im * tau / 2 * hj)
      push!(gates, Gj)
    end
    append!(gates, reverse(gates))

    psi = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    phi = deepcopy(psi)
    c = div(N, 2)

    #
    # Evolve using TEBD
    # 

    Nsteps = convert(Int, ceil(abs(ttotal / tau)))
    Sz1 = zeros(Nsteps)
    En1 = zeros(Nsteps)
    Sz2 = zeros(Nsteps)
    En2 = zeros(Nsteps)

    for step in 1:Nsteps
      psi = apply(gates, psi; cutoff)
      #normalize!(psi)

      nsite = (step <= 3 ? 2 : 1)
      phi = tdvp(
        H,
        -tau * im,
        phi;
        nsweeps=1,
        cutoff,
        nsite,
        normalize=true,
        exponentiate_krylovdim=15,
      )

      Sz1[step] = expect(psi, "Sz"; sites=c:c)[1]
      Sz2[step] = expect(phi, "Sz"; sites=c:c)[1]
      En1[step] = real(inner(psi', H, psi))
      En2[step] = real(inner(phi', H, phi))
    end

    #
    # Evolve using TDVP
    # 
    struct TDVPObserver <: AbstractObserver end

    Sz2 = zeros(Nsteps)
    En2 = zeros(Nsteps)
    function ITensors.measure!(obs::TDVPObserver; sweep, bond, half_sweep, psi, kwargs...)
      if bond == 1 && half_sweep == 2
        Sz2[sweep] = expect(psi, "Sz"; sites=c)
        En2[sweep] = real(inner(psi', H, psi))
      end
    end

    phi = productMPS(s, n -> isodd(n) ? "Up" : "Dn")

    phi = tdvp(
      H,
      -im * ttotal,
      phi;
      time_step=-im * tau,
      cutoff,
      normalize=false,
      (observer!)=TDVPObserver(),
      root_vertex=N, # defaults to 1, which breaks observer equality
    )

    @test norm(Sz1 - Sz2) < 1e-3
    @test norm(En1 - En2) < 1e-3
  end

  @testset "Imaginary Time Evolution" for reverse_step in [true, false]
    N = 10
    cutoff = 1e-12
    tau = 1.0
    ttotal = 50.0

    s = siteinds("S=1/2", N)

    os = OpSum()
    for j in 1:(N - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end

    H = MPO(os, s)

    psi = randomMPS(s; linkdims=2)
    psi2 = deepcopy(psi)
    trange = 0.0:tau:ttotal
    for (step, t) in enumerate(trange)
      nsite = (step <= 10 ? 2 : 1)
      psi = tdvp(
        H, -tau, psi; cutoff, nsite, reverse_step, normalize=true, exponentiate_krylovdim=15
      )
      #Different backend solvers, default solver_backend = "applyexp"
      psi2 = tdvp(
        H,
        -tau,
        psi2;
        cutoff,
        nsite,
        reverse_step,
        normalize=true,
        exponentiate_krylovdim=15,
        solver_backend="exponentiate",
      )
    end

    @test psi ≈ psi2 rtol = 1e-6

    en1 = inner(psi', H, psi)
    en2 = inner(psi2', H, psi2)
    @test en1 < -4.25
    @test en1 ≈ en2
  end

  @testset "Observers" begin
    N = 10
    cutoff = 1e-12
    tau = 0.1
    ttotal = 1.0

    s = siteinds("S=1/2", N; conserve_qns=true)

    os = OpSum()
    for j in 1:(N - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end
    H = MPO(os, s)

    c = div(N, 2)

    #
    # Using the ITensors observer system
    # 
    struct TDVPObserver <: AbstractObserver end

    Nsteps = convert(Int, ceil(abs(ttotal / tau)))
    Sz1 = zeros(Nsteps)
    En1 = zeros(Nsteps)
    function ITensors.measure!(obs::TDVPObserver; sweep, bond, half_sweep, psi, kwargs...)
      if bond == 1 && half_sweep == 2
        Sz1[sweep] = expect(psi, "Sz"; sites=c)
        En1[sweep] = real(inner(psi', H, psi))
      end
    end

    psi1 = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    tdvp(
      H,
      -im * ttotal,
      psi1;
      time_step=-im * tau,
      cutoff,
      normalize=false,
      (observer!)=TDVPObserver(),
      root_vertex=N, # defaults to 1, which breaks observer equality
    )

    #
    # Using Observers.jl
    # 

    function measure_sz(; psi, bond, half_sweep)
      if bond == 1 && half_sweep == 2
        return expect(psi, "Sz"; sites=c)
      end
      return nothing
    end

    function measure_en(; psi, bond, half_sweep)
      if bond == 1 && half_sweep == 2
        return real(inner(psi', H, psi))
      end
      return nothing
    end

    function identity_info(; info)
      return info
    end

    obs = Observer("Sz" => measure_sz, "En" => measure_en, "info" => identity_info)

    step_measure_sz(; psi) = expect(psi, "Sz"; sites=c)

    step_measure_en(; psi) = real(inner(psi', H, psi))

    step_obs = Observer("Sz" => step_measure_sz, "En" => step_measure_en)

    psi2 = MPS(s, n -> isodd(n) ? "Up" : "Dn")
    tdvp(
      H,
      -im * ttotal,
      psi2;
      time_step=-im * tau,
      cutoff,
      normalize=false,
      (observer!)=obs,
      (step_observer!)=step_obs,
      root_vertex=N, # defaults to 1, which breaks observer equality
    )

    Sz2 = results(obs)["Sz"]
    En2 = results(obs)["En"]
    infos = results(obs)["info"]

    Sz2_step = results(step_obs)["Sz"]
    En2_step = results(step_obs)["En"]

    @test Sz1 ≈ Sz2
    @test En1 ≈ En2
    @test Sz1 ≈ Sz2_step
    @test En1 ≈ En2_step
    @test all(x -> x.converged == 1, infos)
    @test length(values(infos)) == 180
  end
end

@testset "Tree TDVP" begin
  @testset "Basic TDVP" begin
    cutoff = 1e-12

    tooth_lengths = fill(2, 3)
    root_vertex = (3, 2)
    c = named_comb_tree(tooth_lengths)
    s = siteinds("S=1/2", c)

    os = ITensorNetworks.heisenberg(c)

    H = TTNO(os, s)

    ψ0 = normalize!(randomTTNS(s; link_space=10))

    # Time evolve forward:
    ψ1 = tdvp(H, -0.1im, ψ0; nsweeps=1, cutoff, nsite=1)

    @test ψ1 ≈ tdvp(-0.1im, H, ψ0; nsweeps=1, cutoff, nsite=1)
    @test ψ1 ≈ tdvp(H, ψ0, -0.1im; nsweeps=1, cutoff, nsite=1)

    @test norm(ψ1) ≈ 1.0

    ## Should lose fidelity:
    #@test abs(inner(ψ0,ψ1)) < 0.9

    # Average energy should be conserved:
    @test real(inner(ψ1', H, ψ1)) ≈ inner(ψ0', H, ψ0)

    # Time evolve backwards:
    ψ2 = tdvp(H, +0.1im, ψ1; nsweeps=1, cutoff)

    @test norm(ψ2) ≈ 1.0

    # Should rotate back to original state:
    @test abs(inner(ψ0, ψ2)) > 0.99
  end

  @testset "TDVP: Sum of Hamiltonians" begin
    cutoff = 1e-10

    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)
    s = siteinds("S=1/2", c)

    os1 = OpSum()
    for e in edges(c)
      os1 += 0.5, "S+", src(e), "S-", dst(e)
      os1 += 0.5, "S-", src(e), "S+", dst(e)
    end
    os2 = OpSum()
    for e in edges(c)
      os2 += "Sz", src(e), "Sz", dst(e)
    end

    H1 = TTNO(os1, s)
    H2 = TTNO(os2, s)
    Hs = [H1, H2]

    ψ0 = normalize!(randomTTNS(s; link_space=10))

    ψ1 = tdvp(Hs, -0.1im, ψ0; nsweeps=1, cutoff, nsite=1)

    @test ψ1 ≈ tdvp(-0.1im, Hs, ψ0; nsweeps=1, cutoff, nsite=1)
    @test ψ1 ≈ tdvp(Hs, ψ0, -0.1im; nsweeps=1, cutoff, nsite=1)

    @test norm(ψ1) ≈ 1.0

    ## Should lose fidelity:
    #@test abs(inner(ψ0,ψ1)) < 0.9

    # Average energy should be conserved:
    @test real(sum(H -> inner(ψ1', H, ψ1), Hs)) ≈ sum(H -> inner(ψ0', H, ψ0), Hs)

    # Time evolve backwards:
    ψ2 = tdvp(Hs, +0.1im, ψ1; nsweeps=1, cutoff)

    @test norm(ψ2) ≈ 1.0

    # Should rotate back to original state:
    @test abs(inner(ψ0, ψ2)) > 0.99
  end

  @testset "Custom solver in TDVP" begin
    cutoff = 1e-12

    tooth_lengths = fill(2, 3)
    root_vertex = (3, 2)
    c = named_comb_tree(tooth_lengths)
    s = siteinds("S=1/2", c)

    os = ITensorNetworks.heisenberg(c)

    H = TTNO(os, s)

    ψ0 = normalize!(randomTTNS(s; link_space=10))

    function solver(PH, t, psi0; kwargs...)
      solver_kwargs = (;
        ishermitian=true, tol=1e-12, krylovdim=30, maxiter=100, verbosity=0, eager=true
      )
      psi, info = exponentiate(PH, t, psi0; solver_kwargs...)
      return psi, info
    end

    ψ1 = tdvp(solver, H, -0.1im, ψ0; cutoff, nsite=1)

    @test ψ1 ≈ tdvp(solver, -0.1im, H, ψ0; cutoff, nsite=1)
    @test ψ1 ≈ tdvp(solver, H, ψ0, -0.1im; cutoff, nsite=1)

    @test norm(ψ1) ≈ 1.0

    ## Should lose fidelity:
    #@test abs(inner(ψ0,ψ1)) < 0.9

    # Average energy should be conserved:
    @test real(inner(ψ1', H, ψ1)) ≈ inner(ψ0', H, ψ0)

    # Time evolve backwards:
    ψ2 = tdvp(H, +0.1im, ψ1; cutoff)

    @test norm(ψ2) ≈ 1.0

    # Should rotate back to original state:
    @test abs(inner(ψ0, ψ2)) > 0.99
  end

  @testset "Accuracy Test" begin
    tau = 0.1
    ttotal = 1.0
    cutoff = 1e-12

    tooth_lengths = fill(2, 3)
    root_vertex = (3, 2)
    c = named_comb_tree(tooth_lengths)
    s = siteinds("S=1/2", c)

    os = ITensorNetworks.heisenberg(c)
    H = TTNO(os, s)
    HM = contract(H)

    Ut = exp(-im * tau * HM)

    psi = TTNS(ComplexF64, s, v -> iseven(sum(isodd.(v))) ? "Up" : "Dn")
    psix = contract(psi)

    Sz_tdvp = Float64[]
    Sz_exact = Float64[]

    c = (2, 1)
    Szc = op("Sz", s[c])

    Nsteps = Int(ttotal / tau)
    for step in 1:Nsteps
      psix = noprime(Ut * psix)
      psix /= norm(psix)

      psi = tdvp(
        H,
        -im * tau,
        psi;
        cutoff,
        normalize=false,
        solver_tol=1e-12,
        solver_maxiter=500,
        solver_krylovdim=25,
      )
      push!(Sz_tdvp, real(expect("Sz", psi; sites=[c])[c]))
      push!(Sz_exact, real(scalar(dag(prime(psix, s[c])) * Szc * psix)))
      F = abs(scalar(dag(psix) * contract(psi)))
    end

    @test norm(Sz_tdvp - Sz_exact) < 1e-5 # broken when rebasing local draft on remote main, fix this
  end

  # TODO: apply gates in ITensorNetworks

  @testset "TEBD Comparison" begin
    cutoff = 1e-12
    maxdim = typemax(Int)
    tau = 0.1
    ttotal = 1.0

    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)
    s = siteinds("S=1/2", c)

    os = ITensorNetworks.heisenberg(c)
    H = TTNO(os, s)

    gates = ITensor[]
    for e in edges(c)
      s1 = s[src(e)]
      s2 = s[dst(e)]
      hj =
        op("Sz", s1) * op("Sz", s2) +
        1 / 2 * op("S+", s1) * op("S-", s2) +
        1 / 2 * op("S-", s1) * op("S+", s2)
      Gj = exp(-1.0im * tau / 2 * hj)
    push!(gates, Gj)
    end
    append!(gates, reverse(gates))

    psi = TTNS(s, v -> iseven(sum(isodd.(v))) ? "Up" : "Dn")
    phi = copy(psi)
    c = (2, 1)

    #
    # Evolve using TEBD
    # 

    Nsteps = convert(Int, ceil(abs(ttotal / tau)))
    Sz1 = zeros(Nsteps)
    En1 = zeros(Nsteps)
    Sz2 = zeros(Nsteps)
    En2 = zeros(Nsteps)

    for step in 1:Nsteps
      psi = apply(gates, psi; cutoff, maxdim)
      #normalize!(psi)

      nsite = (step <= 3 ? 2 : 1)
      phi = tdvp(
        H, -tau * im, phi; nsweeps=1, cutoff, nsite, normalize=true, exponentiate_krylovdim=15
      )

      Sz1[step] = real(expect("Sz", psi; sites=[c])[c])
      Sz2[step] = real(expect("Sz", phi; sites=[c])[c])
      En1[step] = real(inner(psi', H, psi))
      En2[step] = real(inner(phi', H, phi))
    end

    #
    # Evolve using TDVP
    # 
    struct TDVPObserver <: AbstractObserver end

    Sz2 = zeros(Nsteps)
    En2 = zeros(Nsteps)
    function ITensors.measure!(obs::TDVPObserver; sweep, pos, half_sweep, psi, kwargs...)
      if last(pos) == (1, 2) && half_sweep == 2
        Sz2[sweep] = real(expect("Sz", psi; sites=[c])[c])
        En2[sweep] = real(inner(psi', H, psi))
      end
    end

    phi = TTNS(s, v -> iseven(sum(isodd.(v))) ? "Up" : "Dn")

    phi = tdvp(
      H,
      -im * ttotal,
      phi;
      time_step=-im * tau,
      cutoff,
      normalize=false,
      (observer!)=TDVPObserver(),
      root_vertex=(3, 2),
    )

    @test norm(Sz1 - Sz2) < 5e-3
    @test norm(En1 - En2) < 5e-3
  end

  @testset "Imaginary Time Evolution" for reverse_step in [true, false]
    cutoff = 1e-12
    tau = 1.0
    ttotal = 50.0

    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)
    s = siteinds("S=1/2", c)

    os = ITensorNetworks.heisenberg(c)
    H = TTNO(os, s)

    psi = normalize!(randomTTNS(s; link_space=2))

    trange = 0.0:tau:ttotal
    for (step, t) in enumerate(trange)
      nsite = (step <= 10 ? 2 : 1)
      psi = tdvp(
        H, -tau, psi; cutoff, nsite, reverse_step, normalize=true, exponentiate_krylovdim=15
      )
    end

    @test inner(psi', H, psi) < -2.47
  end

  # TODO: verify quantum number suport in ITensorNetworks

  # @testset "Observers" begin
  #   cutoff = 1e-12
  #   tau = 0.1
  #   ttotal = 1.0

  #   tooth_lengths = fill(2, 3)
  #   c = named_comb_tree(tooth_lengths)
  #   s = siteinds("S=1/2", c; conserve_qns=true)

  #   os = ITensorNetworks.heisenberg(c)
  #   H = TTNO(os, s)

  #   c = (2, 2)

  #   #
  #   # Using the ITensors observer system
  #   # 
  #   struct TDVPObserver <: AbstractObserver end

  #   Nsteps = convert(Int, ceil(abs(ttotal / tau)))
  #   Sz1 = zeros(Nsteps)
  #   En1 = zeros(Nsteps)
  #   function ITensors.measure!(obs::TDVPObserver; sweep, bond, half_sweep, psi, kwargs...)
  #     if bond == 1 && half_sweep == 2
  #       Sz1[sweep] = expect("Sz", psi; sites=[c])[c]
  #       En1[sweep] = real(inner(psi', H, psi))
  #     end
  #   end

  #   psi1 = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
  #   tdvp(
  #     H,
  #     -im * ttotal,
  #     psi1;
  #     time_step=-im * tau,
  #     cutoff,
  #     normalize=false,
  #     (observer!)=TDVPObserver(),
  #     root_vertex=N,
  #   )

  #   #
  #   # Using Observers.jl
  #   # 

  #   function measure_sz(; psi, bond, half_sweep)
  #     if bond == 1  && half_sweep == 2
  #       return expect("Sz", psi; sites=[c])[c]
  #     end
  #     return nothing
  #   end

  #   function measure_en(; psi, bond, half_sweep)
  #     if bond == 1 && half_sweep == 2
  #       return real(inner(psi', H, psi))
  #     end
  #     return nothing
  #   end

  #   obs = Observer("Sz" => measure_sz, "En" => measure_en)

  #   step_measure_sz(; psi) = expect("Sz", psi; sites=[c])[c]

  #   step_measure_en(; psi) = real(inner(psi', H, psi))

  #   step_obs = Observer("Sz" => step_measure_sz, "En" => step_measure_en)

  #   psi2 = MPS(s, n -> isodd(n) ? "Up" : "Dn")
  #   tdvp(
  #     H,
  #     -im * ttotal,
  #     psi2;
  #     time_step=-im * tau,
  #     cutoff,
  #     normalize=false,
  #     (observer!)=obs,
  #     (step_observer!)=step_obs,
  #     root_vertex=N,
  #   )

  #   Sz2 = results(obs)["Sz"]
  #   En2 = results(obs)["En"]

  #   Sz2_step = results(step_obs)["Sz"]
  #   En2_step = results(step_obs)["En"]

  #   @test Sz1 ≈ Sz2
  #   @test En1 ≈ En2
  #   @test Sz1 ≈ Sz2_step
  #   @test En1 ≈ En2_step
  # end  
end

nothing
