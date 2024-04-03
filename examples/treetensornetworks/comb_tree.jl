using NamedGraphs
using ITensors
using ITensorNetworks: TTN
using ITensorUnicodePlots

g = named_comb_tree((5, 2))

@visualize g

s = siteinds("S=1/2", g)
ψ = TTN(s; link_space=3)

@visualize ψ

nothing
