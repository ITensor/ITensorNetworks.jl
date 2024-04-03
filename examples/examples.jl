using ITensorNetworks: IndsNetwork, itensors
using ITensorUnicodePlots: @visualize
using NamedGraphs: named_grid

χ, d = 5, 2
system_dims = (4, 4)
g = named_grid(system_dims)

# Network of indices
is = IndsNetwork(g; link_space=χ, site_space=d)

tn = ITensorNetwork(is)

it = itensors(tn)
@visualize it

nothing
