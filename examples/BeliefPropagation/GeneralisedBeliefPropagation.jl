using ITensors
using ITensorNetworks


include("../peps/utils.jl")
include("../../src/belief_propagation/BeliefPropagationFunctions.jl")



Random.seed!(1384)
chi = 2
n =18
z = 3
g = NamedDimGraph(Graphs.SimpleGraphs.random_regular_graph(n, z), [i for i  = 1:n])
display(g)
s = siteinds("S=1/2", g)
ψ = randomITensorNetwork(s; link_space=chi)


szexact = ITensors.expect("Sz", ψ; cutoff = 1e-10, maxdim = 10)
exactszs = Vector{Float64}([szexact[v] for v in vertices(ψ)])

println("Exact Calculated")

niters = 100
partitionsize = 1
npartitions = trunc(Int, n/partitionsize)
subgraphs, subgraphconns, mts = GBP_form_mts(g, ψ, s, niters, npartitions) 
szapprox = GBP_get_single_site_expec(g, ψ, s, mts, subgraphs, subgraphconns, "Sz")
approxszs = Vector{Float64}([szapprox[v] for v in vertices(ψ)])
err = abs.(approxszs - exactszs)
p = plot([i for i = 1:n], err, label = "Simple Belief Propagation")

niters = 100
partitionsize = 2
npartitions = trunc(Int, n/partitionsize)
subgraphs, subgraphconns, mts = GBP_form_mts(g, ψ, s, niters, npartitions) 
szapprox = GBP_get_single_site_expec(g, ψ, s, mts, subgraphs, subgraphconns, "Sz")
approxszs = Vector{Float64}([szapprox[v] for v in vertices(ψ)])
err = abs.(approxszs - exactszs)
plot!(p, [i for i = 1:n],  err, label = "General Belief Propagation, 2 Site Subgraphs", ylabel = "Sz Error", xlabel = "Site")

display(p)