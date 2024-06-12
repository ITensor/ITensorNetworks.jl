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
  g = named_grid((24, 1); periodic=true)
  g = renamer(g)
  save = true
  L = length(vertices(g))
  h, hl, J = 0.6, 0.2, 1.0
  s = siteinds("S=1/2", g)
  χ = 2
  ψ0 = random_tensornetwork(s; link_space=χ)

  H = ising(s; h, hl, J1=J)
  tnos = opsum_to_tno(s, H)

  energy_calc_fun =
    (tn, bp_cache) -> sum(expect(tn, H; alg="bp", (cache!)=Ref(bp_cache))) / L

  ψfinal, energies = bp_dmrg(ψ0, tnos; no_sweeps=5, energy_calc_fun)
  return final_mags = expect(ψfinal, "Z", ; alg="bp")
end

main()
