module ModelNetworks
using ..ITensorNetworks: IndsNetwork, delta_network, insert_linkinds, itensor
using Graphs: degree, dst, edges, src
using ITensors: commoninds, diag_itensor, inds, noprime
using LinearAlgebra: Diagonal, eigen
using NamedGraphs: NamedGraph

# Build Z of classical Ising model on a given graph at inverse temp beta.
# H = -sum_{(v,v') in edges} sigma^z_v sigma^z_{v'}
# Optional arguments:
#   h: external magnetic field
#   szverts: a list of vertices over which to apply Sz.
#     The resultant network can then be contracted and divided by the actual
#     partition function to get that observable.
# IndsNetwork is assumed to be built from a graph (no site inds) and of link space 2.
function ising_network(
        eltype::Type, s::IndsNetwork, beta::Number; h::Number = 0.0, szverts = nothing
    )
    s = insert_linkinds(s; link_space = 2)
    tn = delta_network(eltype, s)
    if (szverts != nothing)
        for v in szverts
            tn[v] = diag_itensor(eltype[1, -1], inds(tn[v]))
        end
    end
    for edge in edges(tn)
        v1 = src(edge)
        v2 = dst(edge)
        i = commoninds(tn[v1], tn[v2])[1]
        deg_v1 = degree(tn, v1)
        deg_v2 = degree(tn, v2)
        f11 = exp(beta * (1 + h / deg_v1 + h / deg_v2))
        f12 = exp(beta * (-1 + h / deg_v1 - h / deg_v2))
        f21 = exp(beta * (-1 - h / deg_v1 + h / deg_v2))
        f22 = exp(beta * (1 - h / deg_v1 - h / deg_v2))
        q = eltype[f11 f12; f21 f22]
        w, V = eigen(q)
        w = map(sqrt, w)
        sqrt_q = V * Diagonal(w) * inv(V)
        t = itensor(sqrt_q, i, i')
        tn[v1] = tn[v1] * t
        tn[v1] = noprime(tn[v1])
        t = itensor(sqrt_q, i', i)
        tn[v2] = tn[v2] * t
        tn[v2] = noprime(tn[v2])
    end
    return tn
end

function ising_network(s::IndsNetwork, beta::Number; h::Number = 0.0, szverts = nothing)
    return ising_network(typeof(beta), s, beta; h, szverts)
end

function ising_network(
        eltype::Type, g::NamedGraph, beta::Number; h::Number = 0.0, szverts = nothing
    )
    return ising_network(eltype, IndsNetwork(g; link_space = 2), beta; h, szverts)
end

function ising_network(g::NamedGraph, beta::Number; h::Number = 0.0, szverts = nothing)
    return ising_network(eltype(beta), g, beta; h, szverts)
end

# Build the wavefunction whose norm is equal to Z of the classical Ising model.
# s needs to have site indices in this case.
function ising_network_state(eltype::Type, s::IndsNetwork, beta::Number; h::Number = 0.0)
    return ising_network(eltype, s, 0.5 * beta; h)
end

function ising_network_state(eltype::Type, g::NamedGraph, beta::Number; h::Number = 0.0)
    return ising_network(eltype, IndsNetwork(g, 2, 2), 0.5 * beta; h)
end

function ising_network_state(s::IndsNetwork, beta::Number; h::Number = 0.0)
    return ising_network_state(typeof(beta), s, beta; h)
end

function ising_network_state(g::NamedGraph, beta::Number; h::Number = 0.0)
    return ising_network(typeof(beta), IndsNetwork(g, 2, 2), 0.5 * beta; h)
end
end
