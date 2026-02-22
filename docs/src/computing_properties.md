# Computing Properties

## Inner Products and Norms

For general `ITensorNetwork` states, inner products are computed by constructing and
contracting the combined bra–ket network. The default algorithm is **belief propagation**
(`alg="bp"`), which is efficient for large and loopy networks. Use `alg="exact"` for
exact contraction (only practical for small networks or trees).

```julia
z = inner(phi, psi)               # ⟨ϕ|ψ⟩  (belief propagation by default)
z = inner(phi, psi; alg="exact")  # ⟨ϕ|ψ⟩  (exact contraction)
z = inner(phi, A, psi)            # ⟨ϕ|A|ψ⟩
n = norm(psi)                     # √⟨ψ|ψ⟩
```

For numerically large tensor networks where the inner product would overflow, use the
logarithmic variant:

```julia
logz = loginner(phi, psi)         # log(⟨ϕ|ψ⟩) (numerically stable)
```

For `TreeTensorNetwork`, specialised exact methods exploit the tree structure directly
without belief propagation:

```julia
z = inner(x, y)      # ⟨x|y⟩ via DFS contraction
z = inner(y, A, x)   # ⟨y|A|x⟩
n = norm(psi)        # uses ortho_region if available for efficiency
```

```@docs
ITensors.inner(::ITensorNetworks.AbstractITensorNetwork, ::ITensorNetworks.AbstractITensorNetwork)
ITensors.inner(::ITensorNetworks.AbstractITensorNetwork, ::ITensorNetworks.AbstractITensorNetwork, ::ITensorNetworks.AbstractITensorNetwork)
ITensorNetworks.loginner
ITensors.inner(::ITensorNetworks.AbstractTreeTensorNetwork, ::ITensorNetworks.AbstractTreeTensorNetwork)
ITensors.inner(::ITensorNetworks.AbstractTreeTensorNetwork, ::ITensorNetworks.AbstractTreeTensorNetwork, ::ITensorNetworks.AbstractTreeTensorNetwork)
```

## Normalization

`normalize` rescales all tensors in the network by the same factor so that `norm(ψ) ≈ 1`.
For `TreeTensorNetwork`, the normalisation is applied directly at the orthogonality centre.

```julia
psi = normalize(psi)              # exact (default)
psi = normalize(psi; alg = "bp") # belief-propagation (for large loopy networks)
```

```@docs
LinearAlgebra.normalize(::ITensorNetworks.AbstractITensorNetwork)
```

## Expectation Values

### General `ITensorNetwork`

For arbitrary (possibly loopy) tensor networks, expectation values are computed via
**belief propagation** by default. This is approximate for loopy networks but can be made
exact with `alg="exact"` (at exponential cost).

```julia
# Expectation of "Sz" at every vertex
sz = expect(psi, "Sz")

# Selected vertices only
sz = expect(psi, "Sz", [(1,), (3,), (5,)])

# Exact contraction
sz = expect(psi, "Sz"; alg = "exact")
```

When computing multiple operators on the same state, reuse the belief propagation cache
to avoid redundant work:

```julia
using ITensors: Op
sz = expect(psi, Op("Sz", v))    # single-operator form
```

```@docs
ITensorNetworks.expect(::ITensorNetworks.AbstractITensorNetwork, ::String)
ITensorNetworks.expect(::ITensorNetworks.AbstractITensorNetwork, ::String, ::Any)
ITensorNetworks.expect(::ITensorNetworks.AbstractITensorNetwork, ::ITensors.Ops.Op)
```

### `TreeTensorNetwork`

For TTN/MPS states, a specialised exact method exploiting successive orthogonalisations is
available. The operator name is passed as the **first** argument (note the different
argument order from the general form above):

```julia
sz = expect("Sz", psi)                          # all sites
sz = expect("Sz", psi; vertices = [1, 3, 5])   # selected sites
```

This is more efficient than the belief propagation approach for tree-structured networks
because it reuses the orthogonal gauge.

```@docs
ITensorNetworks.expect(::String, ::ITensorNetworks.AbstractTreeTensorNetwork)
```
