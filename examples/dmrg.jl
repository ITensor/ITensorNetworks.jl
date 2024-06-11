using DataGraphs: edge_data, vertex_data
using Dictionaries: Dictionary
using Graphs: nv, vertices
using ITensorMPS: ITensorMPS
using ITensorNetworks:
  AbstractITensorNetwork,
  BeliefPropagationCache,
  ITensorNetworks,
  OpSum,
  ttn,
  apply,
  dmrg,
  inner,
  mpo,
  random_mps,
  random_ttn,
  linkdims,
  siteinds,
  random_tensornetwork,
  maxlinkdim
using ITensorNetworks.ITensorsExtensions: replace_vertices
using ITensorNetworks.ModelHamiltonians: ModelHamiltonians
using ITensors: ITensors, ITensor, expect
using KrylovKit: eigsolve
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using Observers: observer
using NamedGraphs.NamedGraphGenerators: named_grid

using NPZ

include("utils.jl")

Random.seed!(5634)

function main()

  g_str = "HeavyHex"
  pbc = true
  g = g_str == "Chain" ? named_grid((24,1); periodic = pbc) : heavy_hex_lattice_graph(2,2; periodic = pbc)
  g = renamer(g)
  save = false
  chi =1

  N = length(vertices(g))
  g_mps = renamer(named_grid((N,1)))
  h, hl = 1.05, 0.4
  J = 1
  s = siteinds("S=1/2",g_mps)

  os = OpSum()
  for e in edges(g)
    os += J, "Sz", src(e),"Sz",dst(e)
  end
  for v in vertices(g)
    os += h, "Sx", v
    os += hl, "Sz", v
  end 

  H = ttn(os, s)

  psi0 = ttn(random_tensornetwork(s; link_space = chi))
  sweep(; which_sweep, kw...) = which_sweep
  energy(; eigvals, kw...) = eigvals[1]
  region(; which_region_update, sweep_plan, kw...) = first(sweep_plan[which_region_update])
  region_observer! = observer(sweep, region, energy)

  e, psifinal = dmrg(
    H, psi0; nsweeps = 20, maxdim = chi, cutoff= 1e-14, nsites = 1, region_observer!
  )
  energies = (region_observer!.energy) / N
  @show last(energies)
  @show maxlinkdim(psifinal)

  final_mags = expect(psifinal, "Z")
  file_name = "/Users/jtindall/Files/Data/DMRG/"*g_str*"ISINGL$(N)h$(h)hl$(hl)chi$(chi)"
  if pbc
    file_name *= "PBC"
  end
  if save
    npzwrite(file_name*".npz", energies = energies, final_mags = final_mags)
  end

end

main()