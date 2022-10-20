using ITensors
using ITensorNetworks
using Random
using NPZ

include("../peps/utils.jl")
include("../../src/belief_propagation/BeliefPropagationFunctions.jl")

function main()

    Random.seed!(1384)
    chi = 3
    nsmall = 6
    n =nsmall*nsmall
    z = 3
    #g = NamedDimGraph(Graphs.SimpleGraphs.random_regular_graph(n, z), [i for i  = 1:n])
    g = named_grid((nsmall,nsmall))
    s = siteinds("S=1/2", g)
    ψ = randomITensorNetwork(s; link_space=chi)
    println("2D Graph, "*string(n)*" sites")

    psizes = [1,2,3,4]
    niters = [100,20,20,20]

    approxszs = zeros((length(psizes), length(vertices(g))))
    szexact = ITensors.expect("Sz", ψ, s; cutoff = 1e-10, maxdim = 10)
    exactszs = Vector{Float64}([szexact[v] for v in vertices(ψ)])

    count = 1
    for psize in psizes
        niter = niters[count]
        npartitions = trunc(Int, n/psize)
        subgraphs, subgraphconns, mts = GBP_form_mts(g, ψ, s, niter, npartitions) 
        szapprox = GBP_get_single_site_expec(g, ψ, s, mts, subgraphs, subgraphconns, "Sz")
        approxszs[count, :] = Vector{Float64}([szapprox[v] for v in vertices(ψ)])
        err = abs.(approxszs[count, :] - exactszs)
        count += 1
        println("Using "*string(psize)*"Site Subgraphs, Average Error (In Terms of Abs Diff) on Local values of Sz is "*string(sum(err)/n))
    end

    npzwrite("Data/RandomTNCalc2DGraphChi"*string(chi)*"L"*string(n)*".npz", psizes=psizes, exactszs = exactszs, approxszs=approxszs)
end