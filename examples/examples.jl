using ITensors
using ITensorNetworks
using ITensorUnicodePlots

χ, d = 5, 2
dims = (4, 4)
g = set_vertices(grid(dims), dims)

# Network of indices
is = IndsNetwork(g; link_space=χ, site_space=d)

tn = ITensorNetwork(is)

it = itensors(tn)
@visualize it

nothing
