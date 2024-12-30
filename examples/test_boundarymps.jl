using ITensorNetworks: BoundaryMPSCache, BeliefPropagationCache, QuadraticFormNetwork, IndsNetwork, siteinds, ttn, random_tensornetwork,
    partitionedges, messages, update, partition_update, set_messages, message,
    planargraph_partitionedges, update_sequence, switch_messages, orthogonal_mps_update, environment, VidalITensorNetwork, ITensorNetwork, expect,
    default_message_update, contraction_sequence, insert_linkinds,
    partitioned_tensornetwork, default_message
using OMEinsumContractionOrders
using ITensorNetworks.ITensorsExtensions: map_eigvals
using ITensorNetworks.ModelHamiltonians: ising
using ITensors: ITensor, ITensors, Index, OpSum, terms, sites, contract, commonind, replaceind, replaceinds, prime, dag, noncommonind, noncommoninds, inds
using NamedGraphs: NamedGraphs, AbstractGraph, NamedEdge, NamedGraph, vertices, neighbors
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: rem_vertex, add_edges
using NamedGraphs.PartitionedGraphs: PartitionedGraph, partitioned_graph, PartitionVertex, PartitionEdge, unpartitioned_graph,
    partitioned_vertices, which_partition
using LinearAlgebra: normalize
using Graphs: center

using Random

function lieb_lattice_grid(nx::Int64, ny::Int64; periodic=false)
    @assert (!periodic && isodd(nx) && isodd(ny)) || (periodic && iseven(nx) && iseven(ny))
    g = named_grid((nx, ny); periodic)
    for v in vertices(g)
      if iseven(first(v)) && iseven(last(v))
        g = rem_vertex(g, v)
      end
    end
    return g
end

function exact_expect(ψ::ITensorNetwork, ops::Vector{<:String}, vs::Vector)
    s = siteinds(ψ)
    ψIψ = QuadraticFormNetwork(ψ)
    ψOψ = QuadraticFormNetwork(ψ)
    for (op_string, v) in zip(ops, vs)
      ψOψ[(v, "operator")] = ITensors.op(op_string, s[v])
    end
    numer_seq = contraction_sequence(ψOψ; alg="sa_bipartite")
    denom_seq = contraction_sequence(ψIψ; alg="sa_bipartite")
    numer, denom = contract(ψOψ; sequence=numer_seq)[], contract(ψIψ; sequence=denom_seq)[]
    return numer / denom
end
  
function exact_expect(ψ::ITensorNetwork, op_string::String, v)
    return exact_expect(ψ, [op_string], [v])
end

function make_eigs_real(A::ITensor)
    return map_eigvals(x -> real(x), A, first(inds(A)), last(inds(A)); ishermitian=true)
end

Random.seed!(1834)
ITensors.disable_warn_order()


function main()
    L = 4
    #g = lieb_lattice_grid(L, L)
    g = named_hexagonal_lattice_graph(L, L)
    vc = first(center(g))
    s = siteinds("S=1/2", g)
    ψ = random_tensornetwork(ComplexF64, s; link_space = 2)
    bp_update_kwargs = (; maxiter = 50, tol = 1e-14, message_update = ms -> make_eigs_real.(default_message_update(ms)))

    #Run BP first to normalize and put in a stable gauge
    #bpc = BeliefPropagationCache(QuadraticFormNetwork(ψ))
    #bpc = add_edges(bpc, PartitionEdge.(missing_edges(ψ)))
    ψIψ = BeliefPropagationCache(QuadraticFormNetwork(ψ))
    ψIψ = update(ψIψ; bp_update_kwargs...)
    ψ = VidalITensorNetwork(ψ; cache! = Ref(ψIψ), update_cache = false, cache_update_kwargs = (; maxiter = 0))
    ψ = ITensorNetwork(ψ)

    fit_kwargs = (;niters = 50, tolerance = 1e-12)

    ψIψ_bp = BeliefPropagationCache(QuadraticFormNetwork(ψ))

    rs = [2]

    @show exact_expect(ψ, "Z", vc)

    for r in rs
        ψIψ = BoundaryMPSCache(ψIψ_bp; sort_f = v -> first(v), message_rank = r)
        ψIψ = update(ψIψ; mps_fit_kwargs = fit_kwargs)
        ρ = contract(environment(ψIψ, [(vc, "operator")]); sequence = "automatic")
        sz = contract([ρ, ITensors.op("Z", s[vc])])[] /contract([ρ, ITensors.op("I", s[vc])])[]
        @show sz
    end
end

main()