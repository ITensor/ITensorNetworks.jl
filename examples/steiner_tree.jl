using NamedGraphs
using ITensors
using ITensorNetworks
using ITensorUnicodePlots

tn = ITensorNetwork(named_grid((3, 5)); link_space=4)

@visualize tn

terminal_vertices = [(1, 2), (1, 4), (3, 4)]
st = steiner_tree(tn, terminal_vertices)

@show has_edge(st, (1, 2) => (1, 3))
@show has_edge(st, (1, 3) => (1, 4))
@show has_edge(st, (1, 4) => (2, 4))
@show has_edge(st, (2, 4) => (3, 4))
