using NamedGraphs.GraphsExtensions: add_edge, rem_edge
using NamedGraphs: nv, src
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree, named_binary_tree
using ITensors: ITensors, Algorithm, expect, mapprime
using ITensors: dmrg, siteinds
using ITensorNetworks: alternating_update, bp_inserter, bp_extracter, bp_eigsolve_updater, random_tensornetwork, ttn, inner, maxlinkdim, map_inds, combine_linkinds
using ITensorNetworks.ModelHamiltonians: heisenberg, ising
using ITensorNetworks: ITensorNetwork
using Dictionaries
using ITensors: Scaled, Prod, site, which_op, inds, combiner, op, sim
using ITensors.NDTensors: array
using LinearAlgebra

using Random: Random

include("exampleutils.jl")
include("imaginary_time_evo.jl")

ITensors.disable_warn_order()

function smallest_eigvalue(A::AbstractITensorNetwork)
    out = reduce(*, [A[v] for v in vertices(A)])
    out = out * combiner(inds(out; plev = 0))  *combiner(inds(out; plev = 1))
    out = array(out)
    return minimum(LinearAlgebra.eigvals(out))
end

function main()
    L = 4
    h, hl, J = 0.5, 0.1, -1.0
    g = named_grid((L,1); periodic = true)
    #g = heavy_hex_lattice_graph(2,2)
    hs = Dictionary(vertices(g), [h for v in vertices(g)])
    hls = Dictionary(vertices(g), [hl for v in vertices(g)])
    Js = Dictionary(edges(g), [J for e in edges(g)])
    one_site_params = (; hs, hls)
    two_site_params = (; Js)
    s = siteinds("S=1/2", g)
    A = model_tno(s, ising_dictified; two_site_params, one_site_params)
    A2 = first(model_tno_simplified(s, ising_dictified; two_site_params, one_site_params))

    χ, χmax, χTEBDmax  = 1, 3, 3
    nperiods = 10
    dβs = [(25, 0.2/((2^(i-1)))) for i in 1:nperiods]
    nbetas = 200
    Random.seed!(1234)
    cache_update_kwargs = (;maxiter = 25, tol = 1e-10)
    #ψ_init = ITensorNetwork(v -> "↑", s)
    ψ_init = random_tensornetwork(s; link_space = χ)
    H_opsum = ising(s; h, hl, J1 = J)
    e_init = sum(expect(ψ_init, H_opsum; alg = "bp"))
    println("Initial Energy is $e_init")
    #ψ_imag_time = imaginary_time_evo(s, ψ_init, ising_dictified, dβs, nbetas; model_params = (; two_site_params..., one_site_params...), bp_update_kwargs = cache_update_kwargs,
    #apply_kwargs = (; cutoff = 1e-10, maxdim = χTEBDmax))
    #ψ_init = copy(ψ_imag_time)

    inserter_bp_kwargs = (; maxdim = χmax, cutoff = 1e-14, cache_update_kwargs)
    inserter_ttn_kwargs = (; maxdim = χmax)
    updater_kwargs = (; tol = 1e-14, krylovdim = 3, maxiter = 2, verbosity = 0, eager = false)
    nsites, nsweeps = 1, 2

    A = ITensorNetwork[A]
    A2 = ITensorNetwork[A2]
    H_opsum_A2 = ising(underlying_graph(only(A2)); h, hl, J1 = J)
    @show smallest_eigvalue(only(A2))
    @show sum(expect(ψ_init, H_opsum_A2; alg = "bp"))
    #@time e_bp_vectorized, ψ_bp_vectorized = dmrg(A_vec, ψ_init; nsweeps, nsites, updater_kwargs, inserter_kwargs = inserter_bp_kwargs)
    @time e_bp_nonvectorized, ψ_bp_nonvectorized = dmrg(A2, ψ_init; nsweeps, nsites, updater_kwargs, inserter_kwargs = inserter_bp_kwargs)
    @show sum(expect(ψ_bp_nonvectorized, H_opsum_A2; alg = "bp"))
    #@time e_ttn, ψ_ttn = dmrg(ttn(A), ttn(ψ_init); nsweeps, nsites, updater_kwargs, inserter_kwargs = inserter_ttn_kwargs)

    #e_final_vectorized = sum(expect(ψ_bp_vectorized, H_opsum; alg = "bp"))
    e_final_nonvectorized = sum(expect(ψ_bp_nonvectorized, H_opsum; alg = "bp"))

    #@show e_final_vectorized
    @show e_final_nonvectorized
    #@show e_ttn

    #@show expect(ψ_bp_nonvectorized, "Sx")
    @show expect(ψ_bp_nonvectorized, "Sx")
    #@show e_bp
    #@show sum([inner(ψ_bp, A, ψ_bp; alg = "exact")] for A in A_vec) / inner(ψ_bp, ψ_bp; alg = "exact")
end

main()