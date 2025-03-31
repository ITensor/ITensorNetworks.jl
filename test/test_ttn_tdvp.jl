@eval module $(gensym())
using Graphs: dst, edges, src
using ITensors: ITensor, contract, dag, inner, noprime, normalize, prime, scalar
using ITensorNetworks:
  ITensorNetworks,
  OpSum,
  ttn,
  apply,
  expect,
  mpo,
  mps,
  op,
  random_mps,
  random_ttn,
  siteinds,
  tdvp
using ITensorNetworks.ModelHamiltonians: ModelHamiltonians
using LinearAlgebra: norm
using NamedGraphs.NamedGraphGenerators: named_binary_tree, named_comb_tree
using Observers: observer
using StableRNGs: StableRNG
using Test: @testset, @test
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

    H = mpo(os, s)

    rng = StableRNG(1234)
    ψ0 = random_mps(rng, s; link_space=10)

    # Time evolve forward:
    ψ1 = tdvp(H, -0.1im, ψ0; nsweeps=1, cutoff, nsites=1)
    @test norm(ψ1) ≈ 1.0

    ## Should lose fidelity:
    #@test abs(inner(ψ0,ψ1)) < 0.9

    # Average energy should be conserved:
    @test real(inner(ψ1', H, ψ1)) ≈ inner(ψ0', H, ψ0)

    # Time evolve backwards:
    ψ2 = tdvp(
      H,
      +0.1im,
      ψ1;
      nsweeps=1,
      cutoff,
      updater_kwargs=(; krylovdim=20, maxiter=20, tol=1e-8),
    )

    @test norm(ψ2) ≈ 1.0

    # Should rotate back to original state:
    @test abs(inner(ψ0, ψ2)) > 0.99

    # test different ways to specify time-step specifications
    ψa = tdvp(H, -0.1im, ψ0; nsweeps=4, cutoff, nsites=1)
    ψb = tdvp(H, -0.1im, ψ0; time_step=-0.025im, cutoff, nsites=1)
    ψc = tdvp(
      H, -0.1im, ψ0; time_step=[-0.02im, -0.03im, -0.015im, -0.035im], cutoff, nsites=1
    )
    ψd = tdvp(
      H, -0.1im, ψ0; nsweeps=4, time_step=[-0.02im, -0.03im, -0.025im], cutoff, nsites=1
    )
    @test inner(ψa, ψb) ≈ 1.0 rtol = 1e-7
    @test inner(ψa, ψc) ≈ 1.0 rtol = 1e-7
    @test inner(ψa, ψd) ≈ 1.0 rtol = 1e-7
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

    H1 = mpo(os1, s)
    H2 = mpo(os2, s)
    Hs = [H1, H2]

    rng = StableRNG(1234)
    ψ0 = random_mps(rng, s; link_space=10)

    ψ1 = tdvp(Hs, -0.1im, ψ0; nsweeps=1, cutoff, nsites=1)

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

  @testset "Higher-Order TDVP" begin
    N = 10
    cutoff = 1e-12
    order = 4

    s = siteinds("S=1/2", N)

    os = OpSum()
    for j in 1:(N - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end

    H = mpo(os, s)

    rng = StableRNG(1234)
    ψ0 = random_mps(rng, s; link_space=10)

    # Time evolve forward:
    ψ1 = tdvp(H, -0.1im, ψ0; time_step=-0.05im, order, cutoff, nsites=1)

    @test norm(ψ1) ≈ 1.0

    # Average energy should be conserved:
    @test real(inner(ψ1', H, ψ1)) ≈ inner(ψ0', H, ψ0)

    # Time evolve backwards:
    ψ2 = tdvp(H, +0.1im, ψ1; time_step=+0.05im, order, cutoff)

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
    H = mpo(os, s)
    HM = contract(H)

    Ut = exp(-im * tau * HM)

    state = mps(n -> isodd(n) ? "Up" : "Dn", s)
    psi2 = deepcopy(state)
    psix = contract(state)

    Sz_tdvp = Float64[]
    Sz_tdvp2 = Float64[]
    Sz_exact = Float64[]

    c = div(N, 2)
    Szc = op("Sz", s[c])

    Nsteps = Int(ttotal / tau)
    for step in 1:Nsteps
      psix = noprime(Ut * psix)
      psix /= norm(psix)

      state = tdvp(
        H,
        -im * tau,
        state;
        cutoff,
        normalize=false,
        updater_kwargs=(; tol=1e-12, maxiter=500, krylovdim=25),
      )
      # TODO: What should `expect` output? Right now
      # it outputs a dictionary.
      push!(Sz_tdvp, real(expect("Sz", state; vertices=[c])[c]))

      psi2 = tdvp(
        H,
        -im * tau,
        psi2;
        cutoff,
        normalize=false,
        updater_kwargs=(; tol=1e-12, maxiter=500, krylovdim=25),
        updater=ITensorNetworks.exponentiate_updater,
      )
      # TODO: What should `expect` output? Right now
      # it outputs a dictionary.
      push!(Sz_tdvp2, real(expect("Sz", psi2; vertices=[c])[c]))

      push!(Sz_exact, real(scalar(dag(prime(psix, s[c])) * Szc * psix)))
      F = abs(scalar(dag(psix) * contract(state)))
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

    H = mpo(os, s)

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

    state = mps(n -> isodd(n) ? "Up" : "Dn", s)
    phi = deepcopy(state)
    c = div(N, 2)

    #
    # Evolve using TEBD
    # 

    Nsteps = convert(Int, ceil(abs(ttotal / tau)))
    Sz1 = zeros(Nsteps)
    En1 = zeros(Nsteps)
    #Sz2 = zeros(Nsteps)
    #En2 = zeros(Nsteps)

    for step in 1:Nsteps
      state = apply(gates, state; cutoff)

      nsites = (step <= 3 ? 2 : 1)
      phi = tdvp(
        H,
        -tau * im,
        phi;
        nsweeps=1,
        cutoff,
        nsites,
        normalize=true,
        updater_kwargs=(; krylovdim=15),
      )

      Sz1[step] = real(expect("Sz", state; vertices=[c])[c])
      #Sz2[step] = real(expect("Sz", phi; vertices=[c])[c])
      En1[step] = real(inner(state', H, state))
      #En2[step] = real(inner(phi', H, phi))
    end

    #
    # Evolve using TDVP
    # 

    phi = mps(n -> isodd(n) ? "Up" : "Dn", s)

    obs = observer(
      "Sz" => (; state) -> expect("Sz", state; vertices=[c])[c],
      "En" => (; state) -> real(inner(state', H, state)),
    )

    phi = tdvp(
      H,
      -im * ttotal,
      phi;
      time_step=-im * tau,
      cutoff,
      normalize=false,
      (sweep_observer!)=obs,
      root_vertex=N, # defaults to 1, which breaks observer equality
    )

    Sz2 = obs.Sz
    En2 = obs.En
    @test norm(Sz1 - Sz2) < 1e-3
    @test norm(En1 - En2) < 1e-3
  end

  @testset "Imaginary Time Evolution" for reverse_step in [true, false]
    cutoff = 1e-12
    tau = 1.0
    ttotal = 10.0
    N = 10
    s = siteinds("S=1/2", N)

    os = OpSum()
    for j in 1:(N - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end

    H = mpo(os, s)

    rng = StableRNG(1234)
    state = random_mps(rng, s; link_space=2)
    en0 = inner(state', H, state)
    nsites = [repeat([2], 10); repeat([1], 10)]
    maxdim = 32
    state = tdvp(
      H,
      -ttotal,
      state;
      time_step=-tau,
      maxdim,
      cutoff,
      nsites,
      reverse_step,
      normalize=true,
      updater_kwargs=(; krylovdim=15),
    )
    en1 = inner(state', H, state)
    @test en1 < en0
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
    H = mpo(os, s)

    c = div(N, 2)

    #
    # Using Observers.jl
    # 

    measure_sz(; state) = expect("Sz", state; vertices=[c])[c]
    measure_en(; state) = real(inner(state', H, state))
    sweep_obs = observer("Sz" => measure_sz, "En" => measure_en)

    get_info(; info) = info
    step_measure_sz(; state) = expect("Sz", state; vertices=[c])[c]
    step_measure_en(; state) = real(inner(state', H, state))
    region_obs = observer(
      "Sz" => step_measure_sz, "En" => step_measure_en, "info" => get_info
    )

    state2 = mps(n -> isodd(n) ? "Up" : "Dn", s)
    tdvp(
      H,
      -im * ttotal,
      state2;
      time_step=-im * tau,
      cutoff,
      normalize=false,
      (sweep_observer!)=sweep_obs,
      (region_observer!)=region_obs,
      root_vertex=N, # defaults to 1, which breaks observer equality
    )

    Sz2 = sweep_obs.Sz
    En2 = sweep_obs.En

    Sz2_step = region_obs.Sz
    En2_step = region_obs.En
    infos = region_obs.info

    #
    # Could use ideas of other things to test here
    #

    @test all(x -> x.converged == 1, infos)
  end
end

@testset "Tree TDVP" begin
  @testset "Basic TDVP" for c in [named_comb_tree(fill(2, 3)), named_binary_tree(3)]
    cutoff = 1e-12

    tooth_lengths = fill(4, 4)
    root_vertex = (1, 4)
    c = named_comb_tree(tooth_lengths)
    s = siteinds("S=1/2", c)

    os = ModelHamiltonians.heisenberg(c)

    H = ttn(os, s)

    rng = StableRNG(1234)
    ψ0 = normalize(random_ttn(rng, s))

    # Time evolve forward:
    ψ1 = tdvp(H, -0.1im, ψ0; root_vertex, nsweeps=1, cutoff, nsites=2)
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

    H1 = ttn(os1, s)
    H2 = ttn(os2, s)
    Hs = [H1, H2]

    rng = StableRNG(1234)
    ψ0 = normalize(random_ttn(rng, s; link_space=10))

    ψ1 = tdvp(Hs, -0.1im, ψ0; nsweeps=1, cutoff, nsites=1)

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

  @testset "Accuracy Test" begin
    tau = 0.1
    ttotal = 1.0
    cutoff = 1e-12

    tooth_lengths = fill(2, 3)
    root_vertex = (3, 2)
    c = named_comb_tree(tooth_lengths)
    s = siteinds("S=1/2", c)

    os = ModelHamiltonians.heisenberg(c)
    H = ttn(os, s)
    HM = contract(H)

    Ut = exp(-im * tau * HM)

    state = ttn(ComplexF64, v -> iseven(sum(isodd.(v))) ? "Up" : "Dn", s)
    statex = contract(state)

    Sz_tdvp = Float64[]
    Sz_exact = Float64[]

    c = (2, 1)
    Szc = op("Sz", s[c])

    Nsteps = Int(ttotal / tau)
    for step in 1:Nsteps
      statex = noprime(Ut * statex)
      statex /= norm(statex)

      state = tdvp(
        H,
        -im * tau,
        state;
        cutoff,
        normalize=false,
        updater_kwargs=(; tol=1e-12, maxiter=500, krylovdim=25),
      )
      push!(Sz_tdvp, real(expect("Sz", state; vertices=[c])[c]))
      push!(Sz_exact, real(scalar(dag(prime(statex, s[c])) * Szc * statex)))
      F = abs(scalar(dag(statex) * contract(state)))
    end

    @test norm(Sz_tdvp - Sz_exact) < 1e-5
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

    os = ModelHamiltonians.heisenberg(c)
    H = ttn(os, s)

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

    state = ttn(v -> iseven(sum(isodd.(v))) ? "Up" : "Dn", s)
    phi = copy(state)
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
      state = apply(gates, state; cutoff, maxdim)

      nsites = (step <= 3 ? 2 : 1)
      phi = tdvp(
        H,
        -tau * im,
        phi;
        nsweeps=1,
        cutoff,
        nsites,
        normalize=true,
        updater_kwargs=(; krylovdim=15),
      )

      Sz1[step] = real(expect("Sz", state; vertices=[c])[c])
      Sz2[step] = real(expect("Sz", phi; vertices=[c])[c])
      En1[step] = real(inner(state', H, state))
      En2[step] = real(inner(phi', H, phi))
    end

    #
    # Evolve using TDVP
    # 

    phi = ttn(v -> iseven(sum(isodd.(v))) ? "Up" : "Dn", s)
    obs = observer(
      "Sz" => (; state) -> expect("Sz", state; vertices=[c])[c],
      "En" => (; state) -> real(inner(state', H, state)),
    )
    phi = tdvp(
      H,
      -im * ttotal,
      phi;
      time_step=-im * tau,
      cutoff,
      normalize=false,
      (sweep_observer!)=obs,
      root_vertex=(3, 2),
    )

    @test norm(Sz1 - Sz2) < 5e-3
    @test norm(En1 - En2) < 5e-3
    @test abs.(last(Sz1) - last(obs.Sz)) .< 5e-3
    @test abs.(last(Sz2) - last(obs.Sz)) .< 5e-3
  end

  @testset "Imaginary Time Evolution" for reverse_step in [true, false]
    cutoff = 1e-12
    tau = 1.0
    ttotal = 50.0

    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)
    s = siteinds("S=1/2", c)

    os = ModelHamiltonians.heisenberg(c)
    H = ttn(os, s)

    rng = StableRNG(1234)
    state = normalize(random_ttn(rng, s; link_space=2))

    trange = 0.0:tau:ttotal
    for (step, t) in enumerate(trange)
      nsites = (step <= 10 ? 2 : 1)
      state = tdvp(
        H,
        -tau,
        state;
        cutoff,
        nsites,
        reverse_step,
        normalize=true,
        updater_kwargs=(; krylovdim=15),
      )
    end

    @test inner(state', H, state) < -2.47
  end
end
end
