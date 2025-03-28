using ITensorNetworks: ttn
using NamedGraphs: vertices
using NamedGraphs.NamedGraphGenerators:
  named_grid, named_hexagonal_lattice_graph, named_comb_tree
using ITensors: ssiteinds, random_itensor

g = named_comb_tree((4, 3))
s = siteinds("S=1/2", g)
all_inds = reduce(vcat, [s[v] for v in vertices(g)])
T = random_itensor(all_inds)

t = ttn(T, s; maxdim=2, cutoff=1e-16)
