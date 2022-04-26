include("named_binary_tree.jl")

g = named_binary_tree(3)
@show g
@show g[1, 1, :]
@show g[1, 2, :]

using ITensorNetworks
using ITensorUnicodePlots

@visualize g
