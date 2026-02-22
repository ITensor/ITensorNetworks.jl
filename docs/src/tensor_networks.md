# Tensor Networks

## The `ITensorNetwork` Type

An `ITensorNetwork` is the central data structure of this package. It represents a
collection of [`ITensor`](https://itensor.github.io/ITensors.jl/stable/)s arranged on a
graph, where each edge encodes a shared (contracted) index between the neighboring tensors.

Key facts:

- The underlying graph is a [`NamedGraph`](https://github.com/ITensor/NamedGraphs.jl), so
  vertices can be any hashable Julia value: integers, tuples, strings, etc.
- Each vertex holds exactly one `ITensor`.
- Edges and link indices are either inferred from shared `Index` objects (when constructing
  from a collection of `ITensor`s) or inserted automatically (when constructing from an
  `IndsNetwork`).

## Construction

The most common entry point is an `IndsNetwork` — a graph whose vertices and edges carry
`Index` objects.  Generate site indices with the `siteinds` function which takes a site
type string (such as "S=1/2" or "Electron") and a NamedGraph. The NamedGraph can be 
generated from functions such as `named_grid`, `named_comb_tree`, etc. from the NamedGraphs.jl
`NamedGraphGenerators` module:

```julia
using ITensorNetworks, ITensors, NamedGraphs.NamedGraphGenerators

# 3×3 square-lattice tensor network
g = named_grid((3, 3))
s = siteinds("S=1/2", g)            # one spin-½ Index per vertex

# Zero-initialized, bond dimension 2
ψ = ITensorNetwork(s; link_space = 2)

# Product state — every site in the |↑⟩ state
ψ = ITensorNetwork("Up", s)

# Staggered initialization with a vertex-dependent function
ψ = ITensorNetwork(v -> isodd(sum(v)) ? "Up" : "Dn", s)
```

When you already have `ITensor`s in hand, edges are inferred automatically from shared
indices:

```julia
i, j, k = Index(2,"i"), Index(2,"j"), Index(2,"k")
A, B, C  = ITensor(i,j), ITensor(j,k), ITensor(k)

tn = ITensorNetwork([A, B, C])                     # integer vertices 1, 2, 3
tn = ITensorNetwork(["A","B","C"], [A, B, C])       # named vertices
tn = ITensorNetwork(["A"=>A, "B"=>B, "C"=>C])       # from pairs
```

```@docs; canonical=false
ITensorNetworks.ITensorNetwork
```

## Accessing Data

```julia
ψ[(1,2)]                  # ITensor at vertex (1,2)
ψ[(1,2)] = T             # replace tensor at a vertex
vertices(ψ)              # all vertex labels
edges(ψ)                 # all edges
neighbors(ψ, v)          # neighbouring vertices of v
nv(ψ), ne(ψ)            # vertex / edge counts
siteinds(ψ)              # IndsNetwork of site (physical) indices
linkinds(ψ)              # IndsNetwork of bond (virtual) indices
```

## Adding Two `ITensorNetwork`s

Two networks with the same graph and site indices can be added. The result represents the
quantum state `ψ₁ + ψ₂` and has bond dimension equal to the **sum** of the two input bond
dimensions. Individual bonds of the result can be recompressed with `truncate(tn, edge)`.
For `TreeTensorNetwork`, the no-argument form `truncate(ttn; kwargs...)` sweeps and
recompresses all bonds at once.

```julia
ψ12 = add(ψ1, ψ2)
```

```@docs; canonical=false
ITensorNetworks.add(::ITensorNetworks.AbstractITensorNetwork, ::ITensorNetworks.AbstractITensorNetwork)
```

## Bond Truncation

A single bond (edge) of any `ITensorNetwork` can be truncated by SVD:

```julia
tn = truncate(tn, (1,2) => (1,3))   # truncate the bond between vertices (1,2) and (1,3)
tn = truncate(tn, edge)              # or pass an AbstractEdge directly
```

Truncation parameters (`cutoff`, `maxdim`, `mindim`, …) are forwarded to `ITensors.svd`.
For a `TreeTensorNetwork`, the sweep-based `truncate(ttn; kwargs...)` is usually more
convenient because it recompresses the entire network at once with controlled errors;
see the [Tree Tensor Networks](@ref) page.

```@docs
Base.truncate(::ITensorNetworks.AbstractITensorNetwork, ::Graphs.AbstractEdge)
```
