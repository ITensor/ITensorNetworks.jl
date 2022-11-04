using ITensors
using ITensorNetworks
using ITensorNetworks: identityITensorNetwork, formsubgraphs, partition, flatten_thicken_bonds, flattened_inner_network, construct_initial_mts, update_all_mts, get_single_site_expec
using KaHyPar

function run_BP(g, psiflat, psiflatO, s, mts, subgraphs, subgraphconns, niters)

    println("Initial Guess for sz on site (1,1) is "*string(get_single_site_expec(g, psiflat, psiflatO, s, mts, subgraphs, subgraphconns, (1,1))))
    for i = 1:niters
        mts = update_all_mts(g, psiflat, s, mts, subgraphs, subgraphconns, niters)
        approx_sz = get_single_site_expec(g, psiflat, psiflatO, s, mts, subgraphs, subgraphconns, (1,1))
        println("After iteration "*string(i)*" Belief propagation gives sz on site (1,1) as "*string(approx_sz))
    
    end
    

end

#4x4 GRID
n = 4
g = named_grid((n,n))
s = siteinds("S=1/2", g)

#Random Tensor Network, Flatten it too
psi = randomITensorNetwork(s; link_space=2)
psiflat, combiners = flatten_thicken_bonds(deepcopy(psi))


#Apply Sz to site (1,1) and flatten that
psiflatO = flatten_thicken_bonds(psi; ops=["Sz"], vops = [(1,1)], s =s, combiners = combiners)

#Get the value of sz on (1,1) via exact contraction
actual_sz = ITensors.contract(psiflatO)[1]/ITensors.contract(psiflat)[1]

println("Actual value of Sz on site (1,1) is "*string(actual_sz))

println("First Using Simple Belief Propagation (Each Site is a subgraph)")
subgraphs, subgraphconns = formsubgraphs(g, n*n)
mts = construct_initial_mts(g, psiflat, s, subgraphs, subgraphconns; id_init = 1)
niters = 10

run_BP(g, psiflat, psiflatO, s, mts, subgraphs, subgraphconns, niters)

println("Now Using General Belief Propagation (4 sites form a subgraph)")
subgraphs, subgraphconns = formsubgraphs(g, Int(n*n/4))
mts = construct_initial_mts(g, psiflat, s, subgraphs, subgraphconns; id_init = 1)
niters = 10

run_BP(g, psiflat, psiflatO, s, mts, subgraphs, subgraphconns, niters)


