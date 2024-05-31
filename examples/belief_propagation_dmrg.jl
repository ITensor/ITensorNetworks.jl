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

ITensors.disable_warn_order()

function ising_operator_ring(s::IndsNetwork; h = 0, hl = 0)
    L = length(vertices(s))
    s_tree = rem_edge(s, (1,1) => (L,1))
    g_tree = rem_edge(s, (1,1) => (L,1))
    A = ITensorNetwork(ttn(ising(g_tree; h, hl), s_tree))
    A = insert_linkinds(A, [NamedEdge((1,1) => (L,1))])
    Ap2 = ITensorNetwork(v -> v == (L,1) || v == (1,1) ? Op("Sz") : Op("I"), union_all_inds(s,s'))
    return A + Ap2
end

function ising_operator_sq_ring(s::IndsNetwork; h = 0, hl = 0)
    L = length(vertices(s))
    s_tree = rem_edge(s, (1,1) => (L,1))
    g_tree = rem_edge(s, (1,1) => (L,1))
    A = ITensorNetwork(ttn(ising(g_tree; h, hl), s_tree))
    A = insert_linkinds(A, [NamedEdge((1,1) => (L,1))])
    Ap2 = ITensorNetwork(v -> v == (L,1) || v == (1,1) ? Op("Sz") : Op("I"), union_all_inds(s,s'))
    A = A + Ap2

    Ap = map_inds(prime, A; links = [])
    Ap = map_inds(sim, Ap; sites = [])
    
    Asq = copy(A)
    for v in vertices(Asq)
        Asq[v] = Ap[v] * A[v]
        Asq[v] = mapprime(Asq[v], 2, 1)
    end

    return combine_linkinds(Asq)
end

function main()
    L = 4
    h, hl = 0.5, 0.1
    g = named_grid((L,1); periodic = false)
    s = siteinds("S=1/2", g)
    A = ITensorNetwork(ttn(ising(g; h, hl), s))
    #A = ising_operator_ring(s; h, hl)
    #Asq = ising_operator_sq_ring(s; h, hl)

    χ, χmax = 2, 4
    Random.seed!(1234)
    ψ_init = ITensorNetwork(v -> "↓", s)
    A_vec = ITensorNetwork[A1, A2]
    H_opsum = ising(s; h)

    e_init = sum(expect(ψ_init, H_opsum; alg = "bp"))
    @show e_init
    #A = ITensorNetwork(ttn(heisenberg(g), s))
    cache_update_kwargs = (;maxiter = 30, tol = 1e-8)
    inserter_bp_kwargs = (; maxdim = χmax, cutoff = 1e-14, cache_update_kwargs)
    inserter_ttn_kwargs = (; maxdim = χmax)
    updater_kwargs = (; tol = 1e-14, krylovdim = 5, maxiter = 2, verbosity = 0, eager = false)
    nsites, nsweeps = 2, 3

    A = ITensorNetwork[A]
    #@time e_bp_vectorized, ψ_bp_vectorized = dmrg(A_vec, ψ_init; nsweeps, nsites, updater_kwargs, inserter_kwargs = inserter_bp_kwargs)
    @time e_bp_nonvectorized, ψ_bp_nonvectorized = dmrg(A, ψ_init; nsweeps, nsites, updater_kwargs, inserter_kwargs = inserter_bp_kwargs)
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