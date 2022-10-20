using ITensors
using ITensorNetworks
using Random
using Statistics
using NPZ

include("../peps/utils.jl")
include("../../src/belief_propagation/BeliefPropagationFunctions.jl")

function get_exact_szsz(beta, g::NamedDimGraph)

    expecs = Any[]
    ψ = PartitionFunctionITensorNetwork(g,beta)
    norm = ITensors.contract(ψ)[1]
    for edge in edges(ψ)
        v = src(edge)
        vp = dst(edge)
        O =  PartitionFunctionITensorNetworkSzSz(g, beta,v,vp)
        e =  ITensors.contract(O)[1]/norm
        push!(expecs, e)
    end
    return expecs

end

function get_approx_szsz(beta, g::NamedDimGraph, npartitions::Int, niters::Int)
    expecs = Any[]
    ψmts = graph_tensor_network(s, g, beta; link_space = 2)
    subgraphs, subgraphconns, mts = GBP_form_mts(g, ψmts, s,  niters, npartitions)
    nsites = length(vertices(g))
    for edge in edges(ψmts)
        v = src(edge)
        vp = dst(edge)
        if(in_same_subgraph(subgraphs, v, vp) || npartitions == nsites)
            e = 4*GBP_get_two_site_expec(g, ψmts, s, mts, subgraphs, subgraphconns, "Sz", "Sz", v,vp)
            push!(expecs, e)
        end
    end
    return expecs, subgraphs
end

function get_phase_diagram(dbeta, nbetas::Int, g::NamedDimGraph, partition_sizes, niters)
    betas = [dbeta*i for i =1:nbetas]
    npartitionsizes = length(partition_sizes)
    approx_expecs = zeros((nbetas, npartitionsizes))
    nsites  = length(vertices(g))
    nedges = length(edges(g))
    exact_expecs = zeros((nbetas, nedges))

    avg_exact_expecs = zeros((nbetas, npartitionsizes))
    ψ = PartitionFunctionITensorNetwork(g,1.0)
    i = 1
    for beta in betas
        exact_expecs[i, :] = get_exact_szsz(beta, g)
        i += 1
    end

    j = 1
    for nparts in partition_sizes
        psize = Int(nsites/nparts)
        niter = niters[j]
        i=1
        for beta in betas
            t_expecs, subgraphs = get_approx_szsz(beta, g,psize, niter)
            approx_expecs[i, j] = mean(t_expecs)
            i_count = 0
            o_count = 0
            for edge in edges(ψ)
                v = src(edge)
                vp = dst(edge)
                if(in_same_subgraph(subgraphs, v, vp) || psize == nsites)
                    avg_exact_expecs[i, j] += exact_expecs[i, i_count + 1]
                    o_count += 1
                end
                i_count += 1
            end
            avg_exact_expecs[i,j] /= o_count

            i += 1
        end
        j += 1
    end

    c = 1
    for psize in partition_sizes
        println("PSize is "*string(psize))
        println("Abs Differences are ")
        display(abs.(approx_expecs[:, c] - avg_exact_expecs[:, c]))
        c += 1
    end

    return betas, approx_expecs, avg_exact_expecs

end


Random.seed!(1384)

n= 24
g = NamedDimGraph(Graphs.SimpleGraphs.random_regular_graph(n, 3), [i for i  = 1:n])
#g = named_grid((n,n))
chi = 2
s = siteinds("S=1/2", g)
niters =[50, 20,20,20]
partition_sizes = [1,2,3,4]

nbetas = 40
dbeta = 0.05

betas, approx_expecs, avg_exact_expecs = get_phase_diagram(dbeta, nbetas, g, partition_sizes, niters)

npzwrite("Data/IsingCalcRandomRegularGraphL"*string(n)*".npz", betas=betas, approx_expecs = approx_expecs, avg_exact_expecs = avg_exact_expecs, partition_sizes=partition_sizes)