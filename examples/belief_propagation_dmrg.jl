using NamedGraphs.GraphsExtensions: add_edge
using NamedGraphs: nv
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree, named_binary_tree
using ITensors: ITensors, Algorithm, expect
using ITensors: dmrg, siteinds
using ITensorNetworks: alternating_update, bp_inserter, bp_extracter, bp_eigsolve_updater, random_tensornetwork, ttn, inner, maxlinkdim
using ITensorNetworks.ModelHamiltonians: heisenberg, ising
using ITensorNetworks: ITensorNetwork

using ITensors: Scaled, Prod, site, which_op

using Random: Random

include("exampleutils.jl")

ITensors.disable_warn_order()

L = 24
h = 2.5
g = named_grid((L,1))
g = add_edge(g, (L,1) => (1,1))
s = siteinds("S=1/2", g)
χ, χmax = 2, 5
Random.seed!(1234)
#ψ_init = ITensorNetwork(v -> "↑", s)
ψ_init = random_tensornetwork(s; link_space = 2)
A = model_tno(s, ising; h)
H_opsum = ising(s; h)

e_init = sum(expect(ψ_init, H_opsum; alg = "bp"))
@show e_init
@show expect(ψ_bp, "Sx")
#A = ITensorNetwork(ttn(heisenberg(g), s))
cache_update_kwargs = (;maxiter = 25, tol = 1e-5)
inserter_bp_kwargs = (; maxdim = χmax, cache_update_kwargs)
inserter_ttn_kwargs = (; maxdim = χmax)
updater_kwargs = (; tol = 1e-14, krylovdim = 5, maxiter = 5, verbosity = 0, eager = false)
nsites, nsweeps = 2, 5

@time e_bp, ψ_bp = dmrg(A, ψ_init; nsweeps, nsites, updater_kwargs, inserter_kwargs = inserter_bp_kwargs)
#@time e_ttn, ψ_ttn = dmrg(ttn(A), ttn(ψ); nsweeps, nsites, updater_kwargs, inserter_kwargs = inserter_ttn_kwargs)

e_final = sum(expect(ψ_bp, H_opsum; alg = "bp"))

@show e_final, e_bp

@show expect(ψ_bp, "Sx")

#@show e_bp
#@show inner(ψ_bp, A, ψ_bp; alg = "exact") / inner(ψ_bp, ψ_bp; alg = "exact")