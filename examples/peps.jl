using ITensors
using Graphs
using NamedGraphs
using ITensorNetworks
using ITensorUnicodePlots

system_dims = (3, 3)
g = named_grid(system_dims)
s = siteinds("S=1/2", g)

ℋ = ITensorNetworks.heisenberg(g)

χ = 5
ψ = ITensorNetwork(s; link_space=χ)

@visualize ψ edge_labels = (; plevs=true)

ψ′ = prime(ψ; sites=[])

ψψ = ψ′ ⊗ ψ

@visualize ψψ edge_labels = (; plevs=true) width = 60 height = 40

#@show siteinds(ψ)
#@show linkinds(ψ)

ψ′ = addtags(ψ, "X"; links=[(1, 1) => (2, 1)], sites=[(2, 2)])
@show linkinds(ψ′, (1, 1) => (2, 1)) == addtags(linkinds(ψ, (1, 1) => (2, 1)), "X")
@show siteinds(ψ′, (2, 2)) == addtags(siteinds(ψ, (2, 2)), "X")
@show siteinds(ψ′, (1, 1)) == siteinds(ψ, (1, 1))

ψ′ = sim(ψ; links=[(1, 1) => (2, 1)])
@show linkinds(ψ′, (1, 1) => (2, 1)) ≠ linkinds(ψ, (1, 1) => (2, 1))
@show linkinds(ψ′, (1, 1) => (1, 2)) == linkinds(ψ, (1, 1) => (1, 2))

nothing
