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

```julia
using ITensorNetworks, ITensors, NamedGraphs.NamedGraphGenerators
using ITensors: OpSum

# Build a Heisenberg Hamiltonian on a comb tree
g  = named_comb_tree((4, 3))
s  = siteinds("S=1/2", g)
os = OpSum()
for e in edges(g)
    os += 0.5, "S+", src(e), "S-", dst(e)
    os += 0.5, "S-", src(e), "S+", dst(e)
    os +=      "Sz", src(e), "Sz", dst(e)
end
H = ttn(os, s)

# Random initial state (normalise first!)
psi0 = normalize(random_ttn(s; link_space = 4))

# Run DMRG
energy, psi = dmrg(H, psi0;
    nsweeps          = 10,
    nsites           = 2,
    factorize_kwargs = (; cutoff = 1e-10, maxdim = 50),
    outputlevel      = 1,
)
```

Key keyword arguments:

| Keyword | Default | Description |
|---------|---------|-------------|
| `nsweeps` | — | Number of sweeps (**required**) |
| `nsites` | `1` | Sites per local update (1 or 2) |
| `factorize_kwargs` | — | Bond truncation options: `cutoff`, `maxdim`, `mindim`, … (**required**) |
| `outputlevel` | `0` | `0`=silent, `1`=per-sweep, `2`=per-region |

```@docs
ITensorNetworks.eigsolve
ITensorNetworks.dmrg
```

## Time Evolution — `time_evolve`

[`time_evolve`](@ref ITensorNetworks.time_evolve) applies the time-evolution operator
`exp(-i H t)` to an initial state using the Time-Dependent Variational Principle (TDVP)
algorithm. Internally each local step is integrated with a Runge–Kutta method.

Pass a vector (or range) of real time points; the state is evolved **incrementally**
between consecutive points, so you can inspect it at intermediate times:

```julia
times = 0.0:0.05:2.0
psi_t = time_evolve(H, times, psi0;
    nsites           = 2,
    order            = 4,
    factorize_kwargs = (; cutoff = 1e-10, maxdim = 50),
    outputlevel      = 1,
)
```

Key keyword arguments:

| Keyword | Default | Description |
|---------|---------|-------------|
| `nsites` | `2` | Sites per local update (1 or 2) |
| `order` | `4` | Runge–Kutta order |
| `factorize_kwargs` | — | Bond truncation: `cutoff`, `maxdim`, … |
| `outputlevel` | `0` | `0`=silent, `1`=per-step summary |

```@docs
ITensorNetworks.time_evolve
```
