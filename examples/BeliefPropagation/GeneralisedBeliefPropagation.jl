using ITensors
using ITensorNetworks
using Random

include("../peps/utils.jl")
include("../../src/belief_propagation/BeliefPropagationFunctions.jl")



Random.seed!(1384)
chi = 2
n =18
z = 3
g = NamedDimGraph(Graphs.SimpleGraphs.random_regular_graph(n, z), [i for i  = 1:n])
s = siteinds("S=1/2", g)
ψ = randomITensorNetwork(s; link_space=chi)

println("Random Regular Graph")

szexact = ITensors.expect("Sz", ψ; cutoff = 1e-10, maxdim = 10)
exactszs = Vector{Float64}([szexact[v] for v in vertices(ψ)])

println("Exact Expectation Values Calculated")

niters = 100
partitionsize = 1
npartitions = trunc(Int, n/partitionsize)
subgraphs, subgraphconns, mts = GBP_form_mts(g, ψ, s, niters, npartitions) 
szapprox = GBP_get_single_site_expec(g, ψ, s, mts, subgraphs, subgraphconns, "Sz")
approxszs = Vector{Float64}([szapprox[v] for v in vertices(ψ)])
err = abs.(approxszs - exactszs)

println("Using 1 Site Subgraphs, Average Error on Local values of Sz is "*string(sum(err)/n))

niters = 25
partitionsize = 2
npartitions = trunc(Int, n/partitionsize)
subgraphs, subgraphconns, mts = GBP_form_mts(g, ψ, s, niters, npartitions) 
szapprox = GBP_get_single_site_expec(g, ψ, s, mts, subgraphs, subgraphconns, "Sz")
approxszs = Vector{Float64}([szapprox[v] for v in vertices(ψ)])
err = abs.(approxszs - exactszs)

println("2 Site Subgraphs, Average Error on Local values of Sz is "*string(sum(err)/n))

n1 = 4
n2 = 4
n = n1*n2
g = named_grid((n1,n2))
s = siteinds("S=1/2", g)
ψ = randomITensorNetwork(s; link_space=chi)


println("2D Grid")

szexact = ITensors.expect("Sz", ψ; cutoff = 1e-10, maxdim = 10)
exactszs = Vector{Float64}([szexact[v] for v in vertices(ψ)])

println("Exact Expectation Values Calculated")

niters = 100
partitionsize = 1
npartitions = trunc(Int, n/partitionsize)
subgraphs, subgraphconns, mts = GBP_form_mts(g, ψ, s, niters, npartitions) 
szapprox = GBP_get_single_site_expec(g, ψ, s, mts, subgraphs, subgraphconns, "Sz")
approxszs = Vector{Float64}([szapprox[v] for v in vertices(ψ)])
err = abs.(approxszs - exactszs)

println("Using 1 Site Subgraphs, Average Error on Local values of Sz is "*string(sum(err)/n))

niters = 25
partitionsize = 4
npartitions = trunc(Int, n/partitionsize)
subgraphs, subgraphconns, mts = GBP_form_mts(g, ψ, s, niters, npartitions) 
szapprox = GBP_get_single_site_expec(g, ψ, s, mts, subgraphs, subgraphconns, "Sz")
approxszs = Vector{Float64}([szapprox[v] for v in vertices(ψ)])
err = abs.(approxszs - exactszs)

println("Using 4 Site Subgraphs, Average Error on Local values of Sz is "*string(sum(err)/n))
