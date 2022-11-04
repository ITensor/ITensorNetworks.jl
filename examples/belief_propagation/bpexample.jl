using ITensors
using ITensorNetworks
using ITensorNetworks: identityITensorNetwork, formsubgraphs, partition, flatten_thicken_bonds, flattened_inner_network, construct_initial_mts, update_all_mts, get_single_site_expec, iterate_single_site_expec
using KaHyPar

#nxn GRID
n = 4
g = named_grid((n,n))
s = siteinds("S=1/2", g)
chi = 3

#Random Tensor Network, Flatten it too
psi = randomITensorNetwork(s; link_space=chi)
psiflat, combiners = flatten_thicken_bonds(deepcopy(psi))

v = (2,2)

#Apply Sz to site v and flatten that
psiflatO = flatten_thicken_bonds(psi; ops=["Sz"], vops = [v], s =s, combiners = combiners)

#Get the value of sz on v via exact contraction
actual_sz = ITensors.contract(psiflatO)[1]/ITensors.contract(psiflat)[1]

println("Actual value of Sz on site "*string(v)*" is "*string(actual_sz))

nsites = 1
println("First "*string(nsites)* " sites form a subgraph")
subgraphs, subgraphconns = formsubgraphs(g, Int(n*n/nsites))
mts = construct_initial_mts(g, psiflat, s, subgraphs, subgraphconns; id_init = 1)
niters = 5

iterate_single_site_expec(g, psiflat, psiflatO, s, mts, subgraphs, subgraphconns, niters, v)

nsites = 4
println("Now "*string(nsites)* " sites form a subgraph")
subgraphs, subgraphconns = formsubgraphs(g, Int(n*n/nsites))
mts = construct_initial_mts(g, psiflat, s, subgraphs, subgraphconns; id_init = 1)

iterate_single_site_expec(g, psiflat, psiflatO, s, mts, subgraphs, subgraphconns, niters, v)


