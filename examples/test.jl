using ITensorNetworks: IndsNetwork, siteinds, ttn
using ITensorNetworks.ModelHamiltonians: ising
using ITensors: Index, OpSum, terms, sites
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.GraphsExtensions: rem_vertex

function filter_terms(H, verts)
    H_new = OpSum()
    for term in terms(H)
        if isempty(filter(v -> v ∈ verts, sites(term)))
            H_new += term
        end
    end
    return H_new
end

g = named_grid((8,1))
s = siteinds("S=1/2", g)
H = ising(s)
H_mod = filter_terms(H, [(4,1)])
ttno = ttn(H_mod, s)