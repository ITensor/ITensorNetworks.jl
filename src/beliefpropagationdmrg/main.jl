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
    g_str = "HeavyHex"
    g = g_str == "Chain" ? named_grid((12,1); periodic = true) : heavy_hex_lattice_graph(2,2; periodic = true)
    g = renamer(g)
    save = true
    L = length(vertices(g))
    h, hl, J = 1.05, 0.4, 1.0
    s = siteinds("S=1/2", g)
    χ = 2
    #ψ0 = ITensorNetwork(v -> "↑", s)
    ψ0 = random_tensornetwork(s; link_space = 2)

    H = ising(s; h, hl, J1 = J)
    tnos = opsum_to_tno(s, H)
    tno = reduce(+, tnos)

    energy_calc_fun = (tn, bp_cache) -> sum(expect(tn, H; alg = "bp", (cache!) = Ref(bp_cache)))/L

    ψfinal, energies = bp_dmrg(ψ0, tnos; no_sweeps = 5, energy_calc_fun)
    final_mags = expect(ψfinal, "Z", ; alg = "bp")
    if save
        npzwrite("/Users/jtindall/Files/Data/BPDMRG/"*g_str*"ISINGL$(L)h$(h)hl$(hl)chi$(χ).npz", energies = energies, final_mags = final_mags)
    end

end


main()