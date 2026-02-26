# Tree Tensor Networks

## Overview

A `TreeTensorNetwork` (alias `TTN`) is an `ITensorNetwork` whose underlying graph is a
**tree** (no cycles). This additional structure enables exact, efficient canonical gauges
via QR decomposition — a key ingredient in variational algorithms such as DMRG and TDVP.

A `TreeTensorNetwork` carries an extra piece of metadata: the `ortho_region`, which
records which vertices currently form the orthogonality centre of the network. Algorithms
update this field as the gauge changes.

**MPS** (matrix product states) are the special case of a `TreeTensorNetwork` on a
1D path graph. The [`mps`](@ref ITensorNetworks.mps) constructor enforces this topology
and provides a convenient interface for 1D calculations.

## Construction

### From an `IndsNetwork` or graph

```julia
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensorNetworks: mps, random_mps, random_ttn, siteinds, ttn

let
# Comb-tree TTN (a popular tree topology for 2D-like systems)
g = named_comb_tree((4, 3))
sites = siteinds("S=1/2", g)

psi = ttn(sites)              # zero-initialised
psi = ttn(v -> "Up", sites)   # product state

# Random, normalised TTN
psi = random_ttn(sites; link_space = 4)

# 1D MPS
s1d = siteinds("S=1/2", 10)
mps_state = mps(v -> "Up", s1d)   # product MPS
mps_state  = random_mps(s1d; link_space = 4)
end
```

```@docs; canonical=false
ITensorNetworks.ttn
ITensorNetworks.mps
ITensorNetworks.random_ttn
ITensorNetworks.random_mps
```

### The `TreeTensorNetwork` type and conversion

The `TreeTensorNetwork` struct wraps an `ITensorNetwork` and records the current
orthogonality region. Use the `TreeTensorNetwork` constructor to convert a plain
`ITensorNetwork` with tree topology into a `TTN`, and `ITensorNetwork` to strip the
gauge metadata when you need a plain network again.

```julia
itn = ITensorNetwork(sites; link_space = 2)
psi = TreeTensorNetwork(itn)               # ITensorNetwork → TTN
itn = ITensorNetwork(psi)                  # TTN → ITensorNetwork
```

```@docs; canonical=false
ITensorNetworks.TreeTensorNetwork
ITensorNetworks.ITensorNetwork(::ITensorNetworks.TreeTensorNetwork)
```

### From a dense `ITensor`

A dense tensor can be decomposed into a TTN by successive QR/SVD factorisations along the
tree edges. Truncation parameters (e.g. `cutoff`, `maxdim`) are forwarded to the
factorisation step.

```julia
g = named_comb_tree((3,1))
sites = siteinds("S=1/2",g)
A  = ITensors.random_itensor(sites[(1,1)], sites[(2,1)], sites[(3,1)])
ttn_A = ttn(A, sites)
```

```@docs
ITensorNetworks.ttn(::ITensors.ITensor, ::ITensorNetworks.IndsNetwork)
```

## Orthogonal Gauge

One of the most powerful features of tree tensor networks is the ability to bring the
network into an **orthogonal gauge** in linear time. When the network is in a gauge
centred on vertex `v`, all tensors away from `v` are isometric with respect to the bond
pointing toward `v`. This makes computing local observables, inner products, and
eigenvalue problems numerically efficient and stable.

The current orthogonality centre is tracked by the `ortho_region` field.

```julia
psi = orthogonalize(psi, v)         # QR-sweep to put ortho centre at vertex v
psi = orthogonalize(psi, [v1, v2])  # two-site centre (for nsites=2 sweeps)

ortho_region(psi)                   # query current ortho region (returns an index set)
psi = set_ortho_region(psi, vs)     # update metadata only, no tensor operations
```

```@docs; canonical=false
ITensorNetworks.orthogonalize
ITensorNetworks.ortho_region
ITensorNetworks.set_ortho_region
```

## Bond Truncation

After algorithms that grow the bond dimension (e.g. addition, subspace expansion), use
`truncate` to recompress the network. For `TreeTensorNetwork` there are two forms:

- **Whole-network recompression** (TTN-specific): `truncate(ttn; kwargs...)` sweeps from
  the leaves to the root, orthogonalising and truncating every bond in sequence. This is
  the preferred form after addition or DMRG expansion.
- **Single-bond truncation** (available for any `ITensorNetwork`): `truncate(tn, edge;
  kwargs...)` truncates one bond by SVD — see the [Tensor Networks](@ref) page.

```julia
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

```julia
psi3 = psi1 + psi2             # or add(psi1, psi2)
psi3 = truncate(psi3; cutoff = 1e-10, maxdim = 50)

2.0 * psi                      # scalar multiplication
psi / norm(psi)                # manual normalisation
```

```@docs; canonical=false
ITensorNetworks.add(::ITensorNetworks.AbstractTreeTensorNetwork, ::ITensorNetworks.AbstractTreeTensorNetwork)
Base.:+(::ITensorNetworks.AbstractTreeTensorNetwork...)
```
