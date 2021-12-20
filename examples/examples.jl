using ITensors
using ITensorNetworks
using ITensorUnicodePlots

χ, d = 5, 2
g = set_vertices(grid((2, 2)), (2, 2))

# Network of indices
is = IndsNetwork(g; link_space=χ, site_space=d)

tn = ITensorNetwork(is)

it = itensors(tn)
@visualize it

nothing
