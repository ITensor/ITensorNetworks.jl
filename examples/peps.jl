using ITensors
using ITensorNetworks
using Graphs
using ITensorUnicodePlots

dims = (3, 3)
g = square_lattice_graph(dims)

function heisenberg(g::AbstractGraph)
  # TODO: os = Sum{Op}()
  os = OpSum()
  for e in edges(g)
    os += 1 / 2, "S⁺", src(e), "S⁻", dst(e)
    os += 1 / 2, "S⁺", src(e), "S⁻", dst(e)
    os += "Sᶻ", src(e), "Sᶻ", dst(e)
  end
  return os
end

ℋ = heisenberg(g)
s = siteinds("S=1/2", g)

χ = 5
ψ = ITensorNetwork(s; link_space=χ)

ψt = itensors(ψ)
@visualize ψt edge_labels = (; plevs=true)

# TODO: Implement priming, tagging, etc.
ψ′ = prime(ψ)

ψ′t = itensors(ψ′)
@visualize ψ′t edge_labels = (; plevs=true)

@show siteinds(ψ)
@show linkinds(ψ)

ψ′ = addtags(ψ, "X"; links=[(1, 1) => (2, 1)], sites=[(2, 2)])
@show linkinds(ψ′, (1, 1) => (2, 1)) == addtags(linkinds(ψ, (1, 1) => (2, 1)), "X")
@show siteinds(ψ′, (2, 2)) == addtags(siteinds(ψ, (2, 2)), "X")
@show siteinds(ψ′, (1, 1)) == siteinds(ψ, (1, 1))

ψ′ = sim(ψ; links=[(1, 1) => (2, 1)])
@show linkinds(ψ′, (1, 1) => (2, 1)) ≠ linkinds(ψ, (1, 1) => (2, 1))
@show linkinds(ψ′, (1, 1) => (1, 2)) == linkinds(ψ, (1, 1) => (1, 2))

nothing
