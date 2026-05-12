# Tree Tensor Networks

## Overview

A `TreeTensorNetwork` (alias `TTN`) is an `ITensorNetwork` whose underlying graph is a
**tree** (no cycles). This additional structure enables exact, efficient canonical gauges
via QR decomposition — a key ingredient in variational algorithms such as DMRG and TDVP.

A `TreeTensorNetwork` carries an extra piece of metadata: the `ortho_region`, which
records which vertices currently form the orthogonality center of the network. Algorithms
update this field as the gauge changes.

**MPS** (matrix product states) are the special case of a `TreeTensorNetwork` on a
1D path graph.

## Construction

### From an `OpSum` (Hamiltonian)

A common way to obtain a Hamiltonian-shaped TTN is to convert an `OpSum` over an
`IndsNetwork` of site indices.

```@docs; canonical=false
ITensorNetworks.TreeTensorNetwork(::ITensors.Ops.OpSum, ::ITensorNetworks.IndsNetwork)
```

### From an existing `ITensorNetwork`

The `TreeTensorNetwork` struct wraps an `ITensorNetwork` and records the current
orthogonality region. Use the `TreeTensorNetwork` constructor to convert a plain
`ITensorNetwork` with tree topology into a `TTN`, and `ITensorNetwork` to strip the
gauge metadata when you need a plain network again.

```@example main
using Graphs: edges, vertices
using ITensorNetworks: ITensorNetwork, TreeTensorNetwork, ortho_region, orthogonalize,
    siteinds
using ITensors: ITensors, Index, random_itensor
using LinearAlgebra: norm
using NamedGraphs: NamedGraph
using NamedGraphs.GraphsExtensions: incident_edges
using NamedGraphs.NamedGraphGenerators: named_comb_tree

# Comb-tree TTN (a popular tree topology for 2D-like systems)
g = NamedGraph(named_comb_tree((3, 2)))
sites = siteinds("S=1/2", g)

# Build a structured `ITensorNetwork` with shared link indices on each edge
χ = 2
links = Dict(e => Index(χ, "Link") for e in edges(g))
tensors = Dict(map(collect(vertices(g))) do v
    site_v = sites[v]
    link_v = [haskey(links, e) ? links[e] : links[reverse(e)] for e in incident_edges(g, v)]
    return v => random_itensor(site_v..., link_v...)
end)
itn = ITensorNetwork(tensors)
psi = TreeTensorNetwork(itn)
```

To strip the gauge metadata back to a plain `ITensorNetwork`:

```@example main
itn_again = ITensorNetwork(psi)  # TTN → ITensorNetwork
```

```@docs; canonical=false
ITensorNetworks.TreeTensorNetwork
```

## Orthogonal Gauge

One of the most powerful features of tree tensor networks is the ability to bring the
network into an **orthogonal gauge** in linear time. When the network is in a gauge
centered on vertex `v`, all tensors away from `v` are isometric with respect to the bond
pointing toward `v`. This makes computing local observables, inner products, and
eigenvalue problems numerically efficient and stable.

The current orthogonality center is tracked by the `ortho_region` field.

```@example main
v = collect(vertices(psi))[1]
v1 = collect(vertices(psi))[1]
v2 = collect(vertices(psi))[2]
vs = [v]
psi = orthogonalize(psi, v)  # QR-sweep to put ortho center at vertex v
psi = orthogonalize(psi, [v1, v2])  # two-site center (for nsites=2 sweeps)
ortho_region(psi)  # query current ortho region (returns an index set)
```

```@docs; canonical=false
ITensorNetworks.orthogonalize
ITensorNetworks.ortho_region
```

## Bond Truncation

After algorithms that grow the bond dimension (e.g. addition, subspace expansion), use
`truncate` to recompress the network. For `TreeTensorNetwork` there are two forms:

- **Whole-network recompression** (TTN-specific): `truncate(ttn; kwargs...)` sweeps from
  the leaves to the root, orthogonalising and truncating every bond in sequence. This is
  the preferred form after addition or DMRG expansion.
- **Single-bond truncation** (available for any `ITensorNetwork`): `truncate(tn, edge;
  kwargs...)` truncates one bond by SVD — see the [ITensor Networks](@ref) page.

```@example main
psi = truncate(psi; cutoff = 1e-10, maxdim = 50)
```

The sweep-based form orthogonalises each bond before truncating it, so truncation errors
are controlled. All keyword arguments accepted by `ITensors.svd` (e.g. `cutoff`, `maxdim`,
`mindim`) are forwarded.

```@docs; canonical=false
Base.truncate(::ITensorNetworks.AbstractTreeTensorNetwork)
```

## Addition and Arithmetic

Two TTNs with the same graph and site indices can be summed. The result has bond
dimension equal to the **sum** of the two inputs, and can be recompressed with `truncate`.

```@example main
psi1, psi2 = psi, psi
psi3 = psi1 + psi2  # or add(psi1, psi2)
psi3 = truncate(psi3; cutoff = 1e-10, maxdim = 50)

2 * psi  # scalar multiplication
psi / norm(psi)  # manual normalisation
```

```@docs; canonical=false
ITensorNetworks.add(::ITensorNetworks.AbstractTreeTensorNetwork, ::ITensorNetworks.AbstractTreeTensorNetwork)
Base.:+(::ITensorNetworks.AbstractTreeTensorNetwork...)
```
