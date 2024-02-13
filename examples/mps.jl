using AbstractTrees
using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using Random
using NamedGraphs

Random.seed!(1234)

g = named_path_graph(4)

s = siteinds("S=1/2", g)

ψ = ITensorNetwork(s; link_space=2)

# randomize
randn!.(vertex_data(ψ))

@visualize ψ

is = IndsNetwork(ψ)
v = vertex_data(is)
e = edge_data(is)

ψ̃ = sim(dag(ψ); sites=[])

@visualize ψ̃

ψψ = ("bra" => ψ̃) ⊗ ("ket" => ψ)

@visualize ψψ

# quasi-optimal contraction sequence
sequence = contraction_sequence(ψψ)

print_tree(sequence)

inner_res = contract(ψψ; sequence)[]

@show inner_res

sub = subgraph(ψψ, [(1, "bra"), (1, "ket"), (2, "bra"), (2, "ket")])

@visualize sub
