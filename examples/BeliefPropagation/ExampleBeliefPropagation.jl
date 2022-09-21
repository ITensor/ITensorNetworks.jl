using ITensors
using ITensorNetworks
using NamedGraphs
using DataGraphs
using MultiDimDictionaries
using Graphs
using Random
using LinearAlgebra
using UnicodePlots

include("BeliefPropagationFunctions.jl")
include("../peps/utils.jl")

Random.seed!(1234)

n =14
z = 3
chi = 2
nupdates = 100

println("One Dimensional Chain")
g = NamedDimGraph(Graphs.SimpleGraphs.grid([n]), [i for i  = 1:n])
s = siteinds("S=1/2", g)

ψ = randomITensorNetwork(s; link_space=chi)
fmts, bmts = form_mts(ψ, g, s, nupdates)


approxszs = take_local_expec_using_mts(ψ, fmts, bmts, g, s, "Sz")
actualszs = ITensors.expect("Sz", ψ; cutoff = 1e-10, maxdim = 50)
actualszs = Vector{Float64}([actualszs[v] for v in vertices(ψ)])
println("1D Chain, Difference in Expec per Site")
display(actualszs - approxszs)

g = NamedDimGraph(Graphs.SimpleGraphs.random_regular_graph(n, z), [i for i  = 1:n])
s = siteinds("S=1/2", g)

ψ = randomITensorNetwork(s; link_space=chi)
fmts, bmts = form_mts(ψ, g, s, nupdates)

approxszs = take_local_expec_using_mts(ψ, fmts, bmts, g, s, "Sz")

actualszs = ITensors.expect("Sz", ψ; cutoff = 1e-10, maxdim = 50)
actualszs = Vector{Float64}([actualszs[v] for v in vertices(ψ)])

println("Random Regular Graph, Difference in Expec per Site")
display(actualszs - approxszs)