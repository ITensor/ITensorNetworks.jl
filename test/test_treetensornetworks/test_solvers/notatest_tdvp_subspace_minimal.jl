using ITensors
using ITensorNetworks
using ITensorNetworks: ModelHamiltonians.tight_binding
using ITensorNetworks: exponentiate_updater, _svd_solve_normal, ttn, tdvp
using ITensorNetworks: local_expand_and_exponentiate_updater, vertices
using ITensorNetworks: two_site_expansion_updater, extract_and_truncate, compose_updaters

#using KrylovKit: exponentiate
using Observers
using Random
using Test
using ITensorGaussianMPS: hopping_operator
using LinearAlgebra: exp, I

"""
Propagate a Gaussian density matrix `c` with a noninteracting Hamiltonian `h` up to time `tau`.
"""
function propagate_noninteracting(h::AbstractMatrix, c::AbstractMatrix, tau::Number)
  U = exp(-im * tau * h)
  return U * c * conj(transpose(U))
end

@testset "MPS TDVP" begin
  @testset "2s vs 1s + local subspace" begin
    ITensors.enable_auto_fermion()
    Random.seed!(1234)
    cutoff = 1e-14
    N = 20
    D = 8
    t = 1.0
    tp = 0.0
    g = ITensorNetworks.named_path_graph(N)
    s = siteinds("Fermion", g; conserve_qns=true)
    os = tight_binding(g; t, tp)
    H = ttn(os, s)
    hmat = hopping_operator(os)
    # get the exact result
    tf = 1.0
    taus = LinRange(0, tf, 2)
    #init=I(N)[:,StatsBase.sample(1:N, div(N,2); replace=false)]
    init = I(N)[:, 1:(div(N, 2))] #domain wall initial condition
    states = i -> i <= div(N, 2) ? "Occ" : "Emp"
    init = conj(init) * transpose(init)
    #res=[]
    res = zeros(N, length(taus))
    for (i, tau) in enumerate(taus)
      res[:, i] = real.(diag(propagate_noninteracting(hmat, init, tau))) #densities only
      #plot!(1:N, real.(diag(last(res))))
    end
    psi = ttn(ITensorNetwork(ComplexF64,states,s;link_space=1))
    psi0 = deepcopy(psi)
    function my_sweep_printer(;which_sweep,state,kwargs...)
      m = maximum(ITensorNetworks.edge_data(ITensorNetworks.linkdims(state)))
      println("Sweep ", which_sweep,", maximum bond dimension: ",m)
    end
    ###compose_updater
    dt = 0.05
    tdvp_kwargs=( ;
      updater=compose_updaters(;expander=two_site_expansion_updater,solver=exponentiate_updater),
      updater_kwargs=(;
      expander_kwargs=(;
        cutoff=cutoff,
        maxdim=D,
        svd_func_expand=ITensorNetworks._svd_solve_normal,
        maxdim_func=ITensorNetworks.default_scale_maxdim,
        cutoff_func=ITensorNetworks.default_scale_cutoff_by_timestep,
        ),
      solver_kwargs=(;
        tol=1E-8
      )
      ),
      extracter=extract_and_truncate,
      extracter_kwargs=(;maxdim=D,cutoff=cutoff),
      sweep_printer=my_sweep_printer,
      time_step=-im * dt,
      reverse_step=true,
      order=2,
      normalize=true,
      maxdim=D,
      cutoff=cutoff,
      outputlevel=1,
      )

    #=
    tdvp_kwargs = (
      time_step=-im * dt,
      reverse_step=true,
      order=2,
      normalize=true,
      maxdim=D,
      cutoff=cutoff,
      outputlevel=1,
      updater_kwargs=(;
        expand_kwargs=(;
          cutoff=cutoff,
          maxdim=D,
          svd_func_expand=ITensorNetworks._svd_solve_normal
        ),
        exponentiate_kwargs=(;),
      ),
    )#,exponentiate_kwargs=(;tol=1e-8)))
    =#
    success = false
    psife = nothing
    #while !success
    #  try
    psife = tdvp(
      H,
      -1im * tf,
      psi;
      nsites=1,
      #updater=ITensorNetworks.local_expand_and_exponentiate_updater,
      tdvp_kwargs...,
    )
    #    success=true
    #  catch
    #    println("trying again")
    #  end
    #  
    #end

    @test inner(psi0, psi) ≈ 1 atol = 1e-12
    #@show maxlinkdim(psife)
    #mag_2s=expect("N",psif)
    mag_exp = expect("N", psife)
    # @test real.([mag_2s[v] for v in vertices(psif)]) ≈ res[:,end] atol=1e-3
    @show maximum(abs.(real.([mag_exp[v] for v in vertices(psife)]) .- res[:, end]))
    @test all(
      i -> i < 5e-3, abs.(real.([mag_exp[v] for v in vertices(psife)]) .- res[:, end])
    )
  end
end
nothing
