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
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensors: OpSum
using ITensorNetworks: dmrg, dst, edges, normalize, random_ttn, siteinds, src, ttn

# Build a Heisenberg Hamiltonian on a comb tree
g  = named_comb_tree((3, 2))
s  = siteinds("S=1/2", g)
H = let h = OpSum()
    for e in edges(g)
        h += 0.5, "S+", src(e), "S-", dst(e)
        h += 0.5, "S-", src(e), "S+", dst(e)
        h +=      "Sz", src(e), "Sz", dst(e)
    end
    ttn(h, s)
end

# Random initial state (normalise first!)
psi0 = normalize(random_ttn(s; link_space = 2))

# Run DMRG
energy, psi = dmrg(H, psi0;
    nsweeps          = 2,
    nsites           = 2,
    factorize_kwargs = (; cutoff = 1e-10, maxdim = 10),
    outputlevel      = 1,
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
