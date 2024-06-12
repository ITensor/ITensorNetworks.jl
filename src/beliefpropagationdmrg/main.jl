using ITensorNetworks: random_tensornetwork
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using NPZ
using ITensors: expect
using Random

include("graphsextensions.jl")
include("tensornetworkoperators.jl")
include("bp_dmrg.jl")

Random.seed!(5634)

function main()
    g_str = "Chain"
    g = g_str == "Chain" ? named_grid((24,1); periodic = true) : heavy_hex_lattice_graph(2,2; periodic = true)
    g = renamer(g)
    save = true
    L = length(vertices(g))
    h, hl, J = 0.6, 0.0, 1.0
    s = siteinds("S=1/2", g)
    χ = 2
    dbetas = [(10, 0.5), (10, 0.25), (20, 0.1)]
    ψ0 = ITensorNetwork(v -> "↑", s)
    ψ0 = imaginary_time_evo(s, ψ0, ising, dbetas; model_params = (; h, hl, J1 = J) ,bp_update_kwargs = (; maxiter = 10, tol = 1e-10), apply_kwargs = (; cutoff = 1e-12, maxdim = χ))
    #ψ0 = random_tensornetwork(s; link_space = χ)

    H = ising(s; h, hl, J1 = J)
    tnos = opsum_to_tno(s, H)

    energy_calc_fun = (tn, bp_cache) -> sum(expect(tn, H; alg = "bp", (cache!) = Ref(bp_cache)))/L

    ψfinal, energies = bp_dmrg(ψ0, tnos; no_sweeps = 5, energy_calc_fun)
    final_mags = expect(ψfinal, "Z", ; alg = "bp")
    @show final_mags
    if save
        npzwrite("/Users/jtindall/Files/Data/BPDMRG/"*g_str*"ISINGL$(L)h$(h)hl$(hl)chi$(χ).npz", energies = energies, final_mags = final_mags)
    end

end


main()