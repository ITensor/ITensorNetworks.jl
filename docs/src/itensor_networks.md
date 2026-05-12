# ITensor Networks

## The `ITensorNetwork` Type

An `ITensorNetwork` is the central data structure of this package. It represents a
collection of [`ITensor`](https://itensor.github.io/ITensors.jl/stable/)s arranged on a
graph, where each edge encodes a shared (contracted) index between the neighboring tensors.

Key facts:

- The underlying graph is a [`NamedGraph`](https://github.com/ITensor/NamedGraphs.jl), so
  vertices can be any hashable Julia value: integers, tuples, strings, etc.
- Each vertex holds exactly one `ITensor`.
- Edges are either inferred from shared `Index` objects (when constructing from a
  collection of `ITensor`s) or taken from a graph passed explicitly alongside the tensors.

## Construction

When you already have `ITensor`s in hand, edges are inferred automatically from shared
indices:

```@example main
using Graphs: edges, ne, neighbors, nv, vertices
using ITensorNetworks: ITensorNetwork, add, linkinds, siteinds
using ITensors: Index, ITensor
using NamedGraphs.NamedGraphGenerators: named_grid

i, j, k = Index(2, "i"), Index(2, "j"), Index(2, "k")
A, B, C = ITensor(i, j), ITensor(j, k), ITensor(k)

tn = ITensorNetwork([A, B, C])  # integer vertices 1, 2, 3
tn = ITensorNetwork(Dict("A" => A, "B" => B, "C" => C))  # named vertices via a Dict
```

If you want to control edges directly — for example to build a structured network on a
prescribed lattice and fill in tensors later — pass a `NamedGraph` along with a
collection of `ITensor`s indexed by vertex. To create a tensor network with shared link
indices on each edge, build the indices once per edge and reuse them at both endpoints:

```@example main
using ITensors: random_itensor
using NamedGraphs: NamedGraph
using NamedGraphs.GraphsExtensions: edgetype, incident_edges

g = NamedGraph(named_grid((3, 3)))
s = siteinds("S=1/2", g)  # one spin-½ site Index per vertex

# One shared link Index per edge; bond dimension χ
χ = 2
links = Dict(e => Index(χ, "Link") for e in edges(g))

# Per-vertex tensor: the site Index plus the link Index of every incident edge
tensors = Dict(map(collect(vertices(g))) do v
    site_v = s[v]
    link_v = [haskey(links, e) ? links[e] : links[reverse(e)] for e in incident_edges(g, v)]
    return v => random_itensor(site_v..., link_v...)
end)

ψ = ITensorNetwork(tensors)
```

Higher-level construction routines (random networks, product states, OpSum-derived
TTNs, etc.) are provided by sibling functions like `TreeTensorNetwork(opsum, sites)`
and the test-only helpers in `test/utils.jl`.

```@docs; canonical=false
ITensorNetworks.ITensorNetwork
```

## Accessing Data

```@example main
v = (1, 2)
T = ψ[v]  # ITensor at vertex (1,2)
ψ[v] = T  # replace tensor at a vertex
vertices(ψ)  # all vertex labels
edges(ψ)  # all edges
neighbors(ψ, v)  # neighbouring vertices of v
nv(ψ), ne(ψ)  # vertex / edge counts
siteinds(ψ)  # IndsNetwork of site (physical) indices
linkinds(ψ)  # IndsNetwork of bond (virtual) indices
```

## Adding Two `ITensorNetwork`s

Two networks with the same graph and site indices can be added. The result represents the
tensor network `ψ₁ + ψ₂` and has bond dimension equal to the **sum** of the two input bond
dimensions. Individual bonds of the result can be recompressed with `truncate(tn, edge)`.
For `TreeTensorNetwork`, the no-argument form `truncate(tn; kwargs...)` sweeps and
recompresses all bonds at once.

```@example main
ψ1, ψ2 = ψ, ψ
ψ12 = add(ψ1, ψ2)
ψ12 = ψ1 + ψ2
```

```@docs; canonical=false
ITensorNetworks.add(::ITensorNetworks.AbstractITensorNetwork, ::ITensorNetworks.AbstractITensorNetwork)
```

## Bond Truncation

A single bond (edge) of any `ITensorNetwork` can be truncated by SVD:

```@example main
edge = (1, 2) => (1, 3)
ψ12 = truncate(ψ12, (1, 2) => (1, 3))  # truncate the bond between vertices (1,2) and (1,3)
ψ12 = truncate(ψ12, edge)  # or pass an AbstractEdge directly
```

Truncation parameters (`cutoff`, `maxdim`, `mindim`, …) are forwarded to `ITensors.svd`.
For a `TreeTensorNetwork`, the sweep-based `truncate(tn; kwargs...)` is usually more
convenient because it recompresses the entire network at once with controlled errors;
see the [Tree Tensor Networks](@ref) page.

```@docs; canonical=false
Base.truncate(::ITensorNetworks.AbstractITensorNetwork, ::Graphs.AbstractEdge)
ITensorNetworks.left_orth
ITensorNetworks.left_orth!
```
