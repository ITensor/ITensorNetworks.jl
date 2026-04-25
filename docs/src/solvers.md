# Solvers

ITensorNetworks.jl provides sweep-based solvers for variational problems on tree tensor
networks. All solvers follow the same high-level pattern:

1. Start from an initial `ITensorNetwork` guess.
2. Sweep over the network, solving a small local problem at each site or pair of sites.
3. After each local solve, truncate the updated bond to control bond dimension growth.
4. Repeat for `nsweeps` sweeps.

## Eigenvalue Problems — `eigsolve` / `dmrg`

[`eigsolve`](@ref ITensorNetworks.eigsolve) finds the lowest eigenvalue and corresponding
eigenvector of an operator (e.g. a Hamiltonian) using a DMRG-like
variational sweep algorithm.
[`dmrg`](@ref ITensorNetworks.dmrg) is an alias for `eigsolve`.

```@example main
using Graphs: vertices
using ITensorNetworks: ITensorNetwork, dmrg, dst, edges, normalize, siteinds, src, ttn
using ITensors: Index, OpSum, random_itensor
using NamedGraphs.GraphsExtensions: incident_edges
using NamedGraphs.NamedGraphGenerators: named_comb_tree

function random_state(g, s; link_space = 2)
    l = Dict(e => Index(link_space, "Link") for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    ts = Dict(
        v => random_itensor(only(s[v]), (l[e] for e in incident_edges(g, v))...)
            for v in vertices(g)
    )
    return ITensorNetwork(ts)
end

# Build a Heisenberg Hamiltonian on a comb tree
g = named_comb_tree((3, 2))
s = siteinds("S=1/2", g)
H = let h = OpSum()
    for e in edges(g)
        h += 0.5, "S+", src(e), "S-", dst(e)
        h += 0.5, "S-", src(e), "S+", dst(e)
        h += "Sz", src(e), "Sz", dst(e)
    end
    ttn(h, s)
end

# Random initial state (normalise first!)
psi0 = normalize(ttn(random_state(g, s)))

# Run DMRG
energy, psi = dmrg(H, psi0;
    nsweeps = 2,
    nsites = 2,
    factorize_kwargs = (; cutoff = 1e-10, maxdim = 10),
    outputlevel = 1,
)
```

```@docs
ITensorNetworks.eigsolve
ITensorNetworks.dmrg
```

## Time Evolution — `time_evolve`

```@docs
ITensorNetworks.time_evolve
```
