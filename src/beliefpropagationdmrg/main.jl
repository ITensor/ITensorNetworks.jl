using ITensorNetworks: random_tensornetwork
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensors: expect
using Random

include("graphsextensions.jl")
include("tensornetworkoperators.jl")
include("bp_dmrg.jl")
include("bp_dmrg_V2.jl")
include("utils.jl")

Random.seed!(5634)

function main()
  g = named_grid((24, 1); periodic=true)
  L = length(vertices(g))
  h, hlongitudinal, J = 0.6, 0.2, 1.0
  s = siteinds("S=1/2", g)
  χ = 3
  ψ0 = random_tensornetwork(s; link_space=χ)

  H = ising(s; h, hl=hlongitudinal, J1=J)
  #tnos = opsum_to_tnos_V2(s, H)
  #tno = reduce(+, tnos)

  energy_calc_fun =
    (tn, bp_cache) -> sum(expect(tn, H; alg="bp", (cache!)=Ref(bp_cache))) / L

  #ψfinal, energies = bp_dmrg(ψ0, tnos; no_sweeps=5, energy_calc_fun)
  ψfinal, energies = bp_dmrg(ψ0, H; no_sweeps=5)
  return final_mags = expect(ψfinal, "Z", ; alg="bp")
end

main()
