using ITensorNetworks: random_tensornetwork
using ITensorNetworks.ModelHamiltonians: ising
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensors: expect
using Random

include("bp_dmrg.jl")
include("utils.jl")

Random.seed!(5634)

function main()
  g = heavy_hex_lattice_graph(2,2 ; periodic = true)
  L = length(vertices(g))
  h, hlongitudinal, J = 0.6, 0.2, 1.0
  s = siteinds("S=1/2", g)
  χ = 2
  ψ0 = random_tensornetwork(s; link_space=χ)

  H = ising(s; h, hl=hlongitudinal, J1=J)

  energy_calc_fun =
    (tn, bp_cache) -> sum(expect(tn, H; alg="bp", (cache!)=Ref(bp_cache))) / L

  ψfinal, energies = bp_dmrg(ψ0, H; no_sweeps=5)
  return final_mags = expect(ψfinal, "Z", ; alg="bp")
end

main()
