using ITensors
using ITensorNetworks
using ITensorNetworks: exponentiate_updater
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

    H = mpo(os, s)

    ψ0 = random_mps(s; internal_inds_space=10)

    # Time evolve forward:
    ψ1 = tdvp(H, -0.1im, ψ0; nsteps=1, cutoff, nsites=1)

    @test norm(ψ1) ≈ 1.0

    ## Should lose fidelity:
    #@test abs(inner(ψ0,ψ1)) < 0.9

    # Average energy should be conserved:
    @test real(inner(ψ1', H, ψ1)) ≈ inner(ψ0', H, ψ0)

    # Time evolve backwards:
    ψ2 = tdvp(H, +0.1im, ψ1; nsteps=1, cutoff)

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

    H1 = mpo(os1, s)
    H2 = mpo(os2, s)
    Hs = [H1, H2]

    ψ0 = random_mps(s; internal_inds_space=10)

    ψ1 = tdvp(Hs, -0.1im, ψ0; nsteps=1, cutoff, nsites=1)

    @test norm(ψ1) ≈ 1.0

    ## Should lose fidelity:
    #@test abs(inner(ψ0,ψ1)) < 0.9

    # Average energy should be conserved:
    @test real(sum(H -> inner(ψ1', H, ψ1), Hs)) ≈ sum(H -> inner(ψ0', H, ψ0), Hs)

    # Time evolve backwards:
    ψ2 = tdvp(Hs, +0.1im, ψ1; nsteps=1, cutoff)

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

    ψ0 = random_mps(s; internal_inds_space=10)

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

    state = mps(s; states=(n -> isodd(n) ? "Up" : "Dn"))
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
        updater=exponentiate_updater,
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

    state = mps(s; states=(n -> isodd(n) ? "Up" : "Dn"))
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
      #normalize!(state)

      nsites = (step <= 3 ? 2 : 1)
      phi = tdvp(
        H,
        -tau * im,
        phi;
        nsteps=1,
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

    phi = mps(s; states=(n -> isodd(n) ? "Up" : "Dn"))

    obs = Observer(
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

    H = mpo(os, s)

    state = random_mps(s; internal_inds_space=2)
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

    en1 = inner(state', H, state)
    @test en1 < -4.25
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
    sweep_obs = Observer("Sz" => measure_sz, "En" => measure_en)

    get_info(; info) = info
    step_measure_sz(; state) = expect("Sz", state; vertices=[c])[c]
    step_measure_en(; state) = real(inner(state', H, state))
    region_obs = Observer(
      "Sz" => step_measure_sz, "En" => step_measure_en, "info" => get_info
    )

    state2 = mps(s; states=(n -> isodd(n) ? "Up" : "Dn"))
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
  @testset "Basic TDVP" begin
    cutoff = 1e-12

    tooth_lengths = fill(2, 3)
    root_vertex = (3, 2)
    c = named_comb_tree(tooth_lengths)
    s = siteinds("S=1/2", c)

    os = ITensorNetworks.heisenberg(c)

    H = TTN(os, s)

    ψ0 = normalize!(random_ttn(s; link_space=10))

    # Time evolve forward:
    ψ1 = tdvp(H, -0.1im, ψ0; nsteps=1, cutoff, nsites=1)

    @test norm(ψ1) ≈ 1.0

    ## Should lose fidelity:
    #@test abs(inner(ψ0,ψ1)) < 0.9

    # Average energy should be conserved:
    @test real(inner(ψ1', H, ψ1)) ≈ inner(ψ0', H, ψ0)

    # Time evolve backwards:
    ψ2 = tdvp(H, +0.1im, ψ1; nsteps=1, cutoff)

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

    H1 = TTN(os1, s)
    H2 = TTN(os2, s)
    Hs = [H1, H2]

    ψ0 = normalize!(random_ttn(s; link_space=10))

    ψ1 = tdvp(Hs, -0.1im, ψ0; nsteps=1, cutoff, nsites=1)

    @test norm(ψ1) ≈ 1.0

    ## Should lose fidelity:
    #@test abs(inner(ψ0,ψ1)) < 0.9

    # Average energy should be conserved:
    @test real(sum(H -> inner(ψ1', H, ψ1), Hs)) ≈ sum(H -> inner(ψ0', H, ψ0), Hs)

    # Time evolve backwards:
    ψ2 = tdvp(Hs, +0.1im, ψ1; nsteps=1, cutoff)

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
    H = TTN(os, s)
    HM = contract(H)

    Ut = exp(-im * tau * HM)

    state = TTN(ComplexF64, s, v -> iseven(sum(isodd.(v))) ? "Up" : "Dn")
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

    os = ITensorNetworks.heisenberg(c)
    H = TTN(os, s)

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

    state = TTN(s, v -> iseven(sum(isodd.(v))) ? "Up" : "Dn")
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
      #normalize!(state)

      nsites = (step <= 3 ? 2 : 1)
      phi = tdvp(
        H,
        -tau * im,
        phi;
        nsteps=1,
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

    phi = TTN(s, v -> iseven(sum(isodd.(v))) ? "Up" : "Dn")
    obs = Observer(
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

    os = ITensorNetworks.heisenberg(c)
    H = TTN(os, s)

    state = normalize!(random_ttn(s; link_space=2))

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

  # TODO: verify quantum number suport in ITensorNetworks

  # @testset "Observers" begin
  #   cutoff = 1e-12
  #   tau = 0.1
  #   ttotal = 1.0

  #   tooth_lengths = fill(2, 3)
  #   c = named_comb_tree(tooth_lengths)
  #   s = siteinds("S=1/2", c; conserve_qns=true)

  #   os = ITensorNetworks.heisenberg(c)
  #   H = TTN(os, s)

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
  #       Sz1[sweep] = expect("Sz", psi; vertices=[c])[c]
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
  #       return expect("Sz", psi; vertices=[c])[c]
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

  #   step_measure_sz(; psi) = expect("Sz", psi; vertices=[c])[c]

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
