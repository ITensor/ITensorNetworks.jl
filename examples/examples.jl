using ITensors
using ITensorNetworks
using ITensorUnicodePlots

χ, d = 5, 2
dims = (4, 4)
vertices = vec([(i, j) for i in 1:dims[1], j in 1:dims[2]])
g = NamedDimGraph(grid(dims), vertices)

# Network of indices
is = IndsNetwork(g; link_space=χ, site_space=d)

tn = ITensorNetwork(is)

it = itensors(tn)
@visualize it

nothing
