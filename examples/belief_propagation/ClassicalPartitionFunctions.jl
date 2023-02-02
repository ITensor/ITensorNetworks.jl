using ITensors
using ITensorNetworks
using Random
using Statistics
using NPZ
using ITensorNetworks:
  ising_network, delta_network, construct_initial_mts, update_all_mts, get_two_site_expec
using Compat
using Metis
using NamedGraphs

function get_exact_szsz(beta::Float64, s::IndsNetwork)
  expecs = Dict{NamedEdge{Tuple},Float64}()
  ψ = ising_network(s, beta)
  norm = ITensors.contract(ψ)[1]
  for edge in edges(ψ)
    v = src(edge)
    vp = dst(edge)
    O = ising_network(s, beta; szverts=[v, vp])
    num = ITensors.contract(O)[1]
    e = num / norm
    expecs[edge] = e
  end
  return expecs
end

function run_over_betas(betas, g, niters::Int64, s::IndsNetwork, iter_history, init_mode)
  ψinit = ising_network(s, 0.0)
  nsites = 4
  mts = construct_initial_mts(ψinit, nsites; init=(I...) -> @compat allequal(I) ? 1 : 0)

  exact_szsz_mean = zeros((length(betas)))
  approx_szsz_mean = zeros((length(betas), niters))
  err = zeros((length(betas)))

  cbeta = 1
  for beta in betas
    println("Beta is " * string(beta))
    exact_szszs = get_exact_szsz(beta, s)
    exact_szsz_mean[cbeta] = mean(getindex.(Ref(exact_szszs), keys(exact_szszs)))
    println(
      "Exact Value is " * string(mean(getindex.(Ref(exact_szszs), keys(exact_szszs))))
    )
    ψ = ising_network(s, beta)
    citer = 1
    if (init_mode != "Adiabatic")
      mts = construct_initial_mts(ψinit, nsites; init=(I...) -> @compat allequal(I) ? 1 : 0)
    end
    for niter in 1:niters
      mts = update_all_mts(deepcopy(ψ), mts, 1)

      if (iter_history == true || niter == niters)
        expecs_full = Dict{NamedEdge{Tuple},Float64}()
        expecs_reduced = Dict{NamedEdge{Tuple},Float64}()
        exact_expecs_reduced = Dict{NamedEdge{Tuple},Float64}()
        for edge in edges(ψ)
          v = src(edge)
          vp = dst(edge)
          ψO = ising_network(s, beta; szverts=[v, vp])
          e = get_two_site_expec(deepcopy(ψ), mts, deepcopy(ψO), v, vp)
          expecs_full[edge] = e
        end

        unrav_szszs = getindex.(Ref(expecs_full), keys(expecs_full))
        approx_szsz_mean[cbeta, citer] = mean(unrav_szszs)
        if (niter == niters)
          err[cbeta] = mean(
            abs.(unrav_szszs - getindex.(Ref(exact_szszs), keys(exact_szszs)))
          )
        end
      end

      citer = citer + 1
    end
    display(err[cbeta])
    display(approx_szsz_mean[cbeta, niters])
    cbeta = cbeta + 1
  end

  return exact_szsz_mean, approx_szsz_mean, err
end

Random.seed!(2054)

n = 4
g = named_grid((n, n))
#g = build_triangular_lattice(n,n)
chi = 2
s = IndsNetwork(g; link_space=2)
betas = reverse([
  0.0,
  0.05,
  0.1,
  0.15,
  0.2,
  0.225,
  0.25,
  0.275,
  0.3,
  0.325,
  0.35,
  0.375,
  0.4,
  0.425,
  0.45,
  0.475,
  0.50,
  0.525,
  0.55,
  0.575,
  0.6,
  0.625,
  0.65,
  0.7,
  0.75,
  0.8,
  0.85,
  0.9,
  0.95,
  1.0,
  1.5,
  2.0,
  3.0,
  4.0,
])
niters = 30

full_history = false
init_mode = "Adiabatic"

exact_szsz_mean, approx_szsz_mean, err = run_over_betas(
  betas, g, niters, s, full_history, init_mode
)

#npzwrite("Data/IsingCalcTriangularGridL"*string(n)*"NPartitions"*string(npartitions)*"History"*string(full_history)*"InitMode"*string(init_mode)*".npz", betas=betas,
#exact_szsz_mean = exact_szsz_mean, approx_szsz_mean = approx_szsz_mean, err=err)
