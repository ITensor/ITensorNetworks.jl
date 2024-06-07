using ITensorNetworks: random_tensornetwork
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using NPZ
using ITensors: expect

include("tensornetworkoperators.jl")
include("bp_dmrg.jl")


function main()
    L = 24
    g = named_grid((L,1); periodic = true)
    h,hl = 0.8, 0.2
    s = siteinds("S=1/2", g)
    χ = 2
    #ψ0 = ITensorNetwork(v -> "↑", s)
    ψ0 = random_tensornetwork(s; link_space = χ)

    H = ising(s; h, hl)
    tnos = opsum_to_tno(s, H)
    tno = reduce(+, tnos)

    ψfinal, energies = bp_dmrg(tnos, ψ0; no_sweeps = 10, vertex_order_func = alt_vertex_order)
    final_mags = expect(ψfinal, "Z", ; alg = "bp")
    npzwrite("/Users/jtindall/Documents/Data/BPDMRG/ISINGL$(L)h$(h)hl$(hl)chi$(χ).npz", energies = energies, final_mags = final_mags)

end


main()