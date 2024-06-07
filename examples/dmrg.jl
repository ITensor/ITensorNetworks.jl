using DataGraphs: edge_data, vertex_data
using Dictionaries: Dictionary
using Graphs: nv, vertices
using ITensorMPS: ITensorMPS
using ITensorNetworks:
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
  random_tensornetwork
using ITensorNetworks.ITensorsExtensions: replace_vertices
using ITensorNetworks.ModelHamiltonians: ModelHamiltonians
using ITensors: ITensors, expect
using KrylovKit: eigsolve
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using Observers: observer
using NamedGraphs.NamedGraphGenerators: named_grid

using NPZ

function main()

  N = 24
  g = heavy_hex_lattice_graph(3,3)
  N = length(vertices(g))
  g_mps = named_grid((N,1))
  g_mps = rename_vertices(v -> (first(v), ), g_mps)
  h, hl = 0.8, 0.2
  J = -1
  s = siteinds("S=1/2",g_mps)
  chi = 10
  pbc = false

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
    H, psi0; nsweeps = 5, maxdim = chi, cutoff= 1e-14, nsites = 1, region_observer!
  )
  energies = region_observer!.energy

  final_mags = expect(psifinal, "Z")
  file_name = "/Users/jtindall/Documents/Data/DMRG/HEAVYHEXISINGL$(N)h$(h)hl$(hl)chi$(chi)"
  if pbc
    file_name *= "PBC"
  end
  npzwrite(file_name*".npz", energies = energies, final_mags = final_mags)

end

main()