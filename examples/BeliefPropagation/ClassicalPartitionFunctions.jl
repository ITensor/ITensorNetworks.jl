using ITensors
using ITensorNetworks
using Random

include("../peps/utils.jl")
include("../../src/belief_propagation/BeliefPropagationFunctions.jl")



Random.seed!(1384)

n= 6
g = NamedDimGraph(Graphs.SimpleGraphs.random_regular_graph(n, 3), [i for i  = 1:n])
g = named_grid((n,n))
chi = 2
s = siteinds("S=1/2", g)

beta = 0.5
ψ = PartitionFunctionITensorNetwork(g,beta)
O =  PartitionFunctionITensorNetworkSzSz(g, beta, (4,4),(5,4))

expec = ITensors.contract(O)[1]/ITensors.contract(ψ)[1]
display(expec)

ψmts = graph_tensor_network(s, g, beta; link_space = 2)
niters =20
npartitions = Int(n*n/1)
subgraphs, subgraphconns, mts = GBP_form_mts(g, ψmts, s,  niters, npartitions)

display(subgraphs)

expec = 4*GBP_get_two_site_expec(g, ψmts, s, mts, subgraphs, subgraphconns, "Sz", "Sz", (4,4),(5,4))
display(expec)