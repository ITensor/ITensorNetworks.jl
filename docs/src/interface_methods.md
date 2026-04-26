# Interface Methods

## To Do Items


## Files to Review

- [X] itensornetwork.jl
- [X] abstractitensornetwork.jl
- [X] apply.jl. Applying gates to ITensorNetworks.
- [X] tebd.jl

- [X] inner.jl
- [X] expect.jl
- [X] normalize.jl

- [X] indextags.jl
- [X] sitetype.jl
- [ ] abstractindsnetwork.jl
- [ ] indsnetwork.jl

- [ ] partitioneditensornetwork.jl
- [ ] specialitensornetworks.jl

- [ ] environment.jl

- [X] graphs.jl
- [ ] edge\_sequences.jl
- [ ] contract.jl
- [ ] contraction\_sequences.jl

- [ ] opsum.jl

- [X] update\_observer.jl
- [X] utils.jl
- [X] visualize.jl

- [ ] caches/abstractbeliefpropagationcache.jl
- [ ] caches/beliefpropagationcache.jl

- [ ] formnetworks/abstractformnetwork.jl
- [ ] formnetworks/bilinearformnetwork.jl
- [ ] formnetworks/linearformnetwork.jl
- [ ] formnetworks/quadraticformnetwork.jl

- [ ] treetensornetworks/abstracttreetensornetwork.jl
- [ ] treetensornetworks/treetensornetwork.jl
- [ ] treetensornetworks/opsum\_to\_ttn/
- [ ] treetensornetworks/projttns/

- [ ] solvers/

Recommended methods for building applications on top of ITensorNetworks.

## ITensorNetwork Constructors

These ITensorNetwork constructor interfaces are foundational to other constructors:

* Default constructor and copy constructor (`itensornetwork.jl`):
  ```julia
  ITensorNetwork{V}()
  ITensorNetwork{V}(tn::ITensorNetwork)
  ```

* From vertex-tensor pairings (`itensornetwork.jl`):
  ```julia
  # Dictionary of vertices => tensors
  ITensorNetwork(ts::AbstractDictionary{<:Any, ITensor})
  ITensorNetwork(ts::AbstractDict{<:Any, ITensor})

  # Vector of `vertex => ITensor` pairs
  ITensorNetwork(ts::AbstractVector{<:Pair{<:Any, ITensor}})

  # Vector of vertices, vector of ITensors
  ITensorNetwork(vertices::AbstractVector, tensors::AbstractVector{ITensor})
  ```

* From a collection of ITensorNetworks. Merges (Kronecker or tensor product) of input networks (`itensornetwork.jl`):
  ```julia
  ITensorNetwork(itns::Vector{ITensorNetwork})
  ```

* From a vector of `ITensor`s, with vertex labels auto-assigned to `eachindex(ts)`.
  Edges are inferred from shared indices (`itensornetwork.jl`):
  ```julia
  ITensorNetwork(ts::AbstractVector{ITensor})
  ```

* From a single `ITensor`. Wraps the tensor in a single-vertex network (`itensornetwork.jl`):
  ```julia
  ITensorNetwork(t::ITensor)
  ```

* From `IndsNetwork`. Initializes ITensors with `undef` storage on each vertex
  of the `IndsNetwork` with the corresponding indices (`itensornetwork.jl`):
  ```julia
  ITensorNetwork(eltype::Type, undef::UndefInitializer, is::IndsNetwork; kwargs...)
  ITensorNetwork(eltype::Type, is::IndsNetwork; kwargs...)
  ITensorNetwork(undef::UndefInitializer, is::IndsNetwork; kwargs...)
  ITensorNetwork(is::IndsNetwork; kwargs...)
  ```

## Analyzing ITensorNetworks

* Indices common to the ITensors on the vertices connected by the edge (`abstractitensornetwork.jl`):
  ```julia
  commoninds(tn::AbstractITensorNetwork, edge)
  linkinds(tn::AbstractITensorNetwork, edge)
  ```

* Collection of tensors neighboring the given vertex (`abstractitensornetwork.jl`):
  ```julia
  neighbor_tensors(tn::AbstractITensorNetwork, vertex)
  ```

* Indices on the source tensor of `edge` that are not shared with the destination tensor.
  For `edge = (v1 => v2)`, returns every index of `tn[v1]` except the bond(s) to `tn[v2]`, that is,
  `v1`'s site index plus its link indices to all of `v1`'s other neighbors. Useful for QR/SVD
  across that bond: these are the indices you keep on the `v1` side (`abstractitensornetwork.jl`):
  ```julia
  uniqueinds(tn::AbstractITensorNetwork, edge::AbstractEdge)
  uniqueinds(tn::AbstractITensorNetwork, edge::Pair)
  # Alias for `uniqueinds`.
  siteinds(tn::AbstractITensorNetwork, edge)
  ```

* Indices on `tn[vertex]` that aren't shared with any neighbor, i.e. the external/site
  indices of that vertex (`abstractitensornetwork.jl`):
  ```julia
  uniqueinds(tn::AbstractITensorNetwork, vertex)
  # Alias for `uniqueinds`.
  siteinds(tn::AbstractITensorNetwork, vertex)
  ```

* Tags on the link index (or indices) associated with `edge` (`abstractitensornetwork.jl`):
  ```julia
  tags(tn::AbstractITensorNetwork, edge)
  ```

* Iterate over the tensors at the given vertices, default all vertices (`abstractitensornetwork.jl`):
  ```julia
  eachtensor(tn::AbstractITensorNetwork, vertices = vertices(tn))
  ```

* Extract the `IndsNetwork` of a tensor network — site indices per vertex and link
  indices per edge (`abstractitensornetwork.jl`):
  ```julia
  IndsNetwork(tn::AbstractITensorNetwork)
  ```

* Collect all site indices (per-vertex) of a network as an `IndsNetwork` or as a flat vector (`abstractitensornetwork.jl`):
  ```julia
  siteinds(tn::AbstractITensorNetwork)
  flatten_siteinds(tn::AbstractITensorNetwork)
  ```

* Collect all link indices (per-edge) of a network as an `IndsNetwork` or as a flat vector (`abstractitensornetwork.jl`):
  ```julia
  linkinds(tn::AbstractITensorNetwork)
  flatten_linkinds(tn::AbstractITensorNetwork)
  ```

* Bond dimension of a single edge, of every edge (as a `DataGraph`), and the maximum
  bond dimension over all edges (`abstractitensornetwork.jl`):
  ```julia
  linkdim(tn::AbstractITensorNetwork{V}, edge::AbstractEdge{V}) where {V}
  linkdims(tn::AbstractITensorNetwork{V}) where {V}
  maxlinkdim(tn::AbstractITensorNetwork)
  ```

## Local Operations on ITensorNetworks

* Contract the tensors at vertices `src(edge)` and `dst(edge)` and store the result in
  `merged_vertex` (which defaults to `dst(edge)`), removing the other vertex (defaults to `src(edge)`) (`abstractitensornetwork.jl`):
  ```julia
  contract(tn::AbstractITensorNetwork, edge::AbstractEdge; merged_vertex = dst(edge))
  contract(tn::AbstractITensorNetwork, edge::Pair; kws...)
  ```

* "Split" an edge index by applying a map to each copy of it on the adjacent ITensors.
  By default the `dst(edge)` copy is primed and the `src(edge)` copy is unchanged (`abstractitensornetwork.jl`):
  ```julia
  split_index(tn::AbstractITensorNetwork, edges_to_split; 
              src_ind_map::Function = identity,
              dst_ind_map::Function = prime)
  ```

* Factorize the bond on `edge` using the default factorization (`abstractitensornetwork.jl`):
  ```julia
  factorize(tn::AbstractITensorNetwork, edge::AbstractEdge; kws...)
  factorize(tn::AbstractITensorNetwork, edge::Pair; kws...)
  ```

* QR-factorize across `edge`, placing the orthogonal factor on `src(edge)` and the remainder
  on `dst(edge)` (`abstractitensornetwork.jl`):
  ```julia
  qr(tn::AbstractITensorNetwork, edge::AbstractEdge; kws...)
  ```

* SVD-factorize across `edge` (`abstractitensornetwork.jl`):
  ```julia
  svd(tn::AbstractITensorNetwork, edge::AbstractEdge; kws...)
  ```

* Truncate the bond on `edge` via SVD; forwards `cutoff`, `maxdim`, `mindim` kwargs (`abstractitensornetwork.jl`):
  ```julia
  truncate(tn::AbstractITensorNetwork, edge::AbstractEdge; kws...)
  ```

## Global Operations on ITensorNetworks

* Tensor product (disjoint union) of two ITensorNetworks (`abstractitensornetwork.jl`):
  ```julia
  ⊗(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
  union(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; kwargs...)
  ```

* Elementwise complex conjugation of every tensor in the network (`abstractitensornetwork.jl`):
  ```julia
  conj(tn::AbstractITensorNetwork)
  ```

* Dagger: conjugate every tensor and prime the appropriate indices (`abstractitensornetwork.jl`):
  ```julia
  dag(tn::AbstractITensorNetwork)
  ```

* Approximate equality of two ITensorNetworks (`abstractitensornetwork.jl`):
  ```julia
  isapprox(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; kws...)
  ```

* Multiply every-vertex tensors by a scalar, multiplied into the first vertex (`abstractitensornetwork.jl`):
  ```julia
  *(c::Number, ψ::AbstractITensorNetwork)
  ```

* Add two ITensorNetworks defined over the same graph; result has summed bond dimensions (`abstractitensornetwork.jl`):
  ```julia
  +(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
  add(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
  ```

* Adjoint: prime all indices of the network (`abstractitensornetwork.jl`):
  ```julia
  adjoint(tn::AbstractITensorNetwork)
  ```

* Rename every vertex `v` of `tn` to `f(v)` (`abstractitensornetwork.jl`):
  ```julia
  rename_vertices(f::Function, tn::AbstractITensorNetwork)
  ```

* Element-type queries and conversions over the whole network (`abstractitensornetwork.jl`):
  ```julia
  scalartype(tn::AbstractITensorNetwork)
  convert_scalartype(eltype::Type{<:Number}, tn::AbstractITensorNetwork)
  complex(tn::AbstractITensorNetwork)
  ```

* Inner product `⟨ϕ|ψ⟩`. Default `alg = "bp"`; `"exact"` builds the bra-ket network and contracts via a sequence (`inner.jl`):
  ```julia
  inner(ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; alg, kwargs...)
  inner(alg::Algorithm, ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...)
  inner(alg::Algorithm"exact", ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; sequence, kwargs...)
  ```

* Matrix element `⟨ϕ|A|ψ⟩` for an operator network `A` (`inner.jl`):
  ```julia
  inner(ϕ::AbstractITensorNetwork, A::AbstractITensorNetwork, ψ::AbstractITensorNetwork; alg, kwargs...)
  inner(alg::Algorithm, ϕ::AbstractITensorNetwork, A::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...)
  inner(alg::Algorithm"exact", ϕ::AbstractITensorNetwork, A::AbstractITensorNetwork, ψ::AbstractITensorNetwork; sequence, kwargs...)
  ```

* Numerically-stable `log(⟨ϕ|ψ⟩)` and `log(⟨ϕ|A|ψ⟩)` (`inner.jl`):
  ```julia
  loginner(ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; alg, kwargs...)
  loginner(ϕ::AbstractITensorNetwork, A::AbstractITensorNetwork, ψ::AbstractITensorNetwork; alg, kwargs...)
  loginner(alg::Algorithm, ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...)
  loginner(alg::Algorithm, ϕ::AbstractITensorNetwork, A::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...)
  loginner(alg::Algorithm"exact", ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...)
  loginner(alg::Algorithm"exact", ϕ::AbstractITensorNetwork, A::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...)
  ```

* Squared norm `⟨ψ|ψ⟩` and norm `√|⟨ψ|ψ⟩|` (`inner.jl`):
  ```julia
  norm_sqr(ψ::AbstractITensorNetwork; kwargs...)
  norm(ψ::AbstractITensorNetwork; kwargs...)
  ```

* Expectation value `⟨ψ|op|ψ⟩ / ⟨ψ|ψ⟩` for a single `Op` (`expect.jl`):
  ```julia
  expect(ψ::AbstractITensorNetwork, op::Op; alg, kwargs...)
  ```

* Local expectation values for the named operator `op` at the given vertices, or at every
  vertex of `ψ`. Returns a `Dictionary` mapping vertex to expectation value (`expect.jl`):
  ```julia
  expect(ψ::AbstractITensorNetwork, op::String, vertices; alg, kwargs...)
  expect(ψ::AbstractITensorNetwork, op::String; alg, kwargs...)
  ```

* Algorithm-specialized dispatches that build a `QuadraticFormNetwork` and either
  share/update a BP cache or contract exactly (`expect.jl`):
  ```julia
  expect(alg::Algorithm, ψ::AbstractITensorNetwork, ops; cache!, update_cache, kwargs...)
  expect(alg::Algorithm"exact", ψ::AbstractITensorNetwork, ops; kwargs...)
  ```

* Single-op evaluator on a pre-built form network, used by the dispatches above (`expect.jl`):
  ```julia
  expect(ψIψ::AbstractFormNetwork, op::Op; kwargs...)
  ```

* Return a copy of `tn` rescaled so that `norm(tn) ≈ 1`, with the rescaling distributed
  evenly across all vertex tensors (`normalize.jl`):
  ```julia
  normalize(tn::AbstractITensorNetwork; alg, kwargs...)
  ```

* Algorithm-specialized dispatches: `"exact"` contracts `⟨ψ|ψ⟩` directly; the generic
  `Algorithm` form uses a cached contraction (e.g. BP) on the inner-product network (`normalize.jl`):
  ```julia
  normalize(alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...)
  normalize(alg::Algorithm, tn::AbstractITensorNetwork; cache!, update_cache, kwargs...)
  ```

## Index Manipulation

* Rewrite every index of a network according to a structural mapping `IndsNetwork => IndsNetwork`
  (site indices per vertex, link indices per edge). The two `IndsNetwork`s must share the
  same underlying graph (`abstractitensornetwork.jl`):
  ```julia
  replaceinds(tn::AbstractITensorNetwork, is_is′::Pair{<:IndsNetwork, <:IndsNetwork})
  ```

* Apply an index-label transformation `f` to every index in the network. Used to implement
  the prime/tag family below (`abstractitensornetwork.jl`):
  ```julia
  map_inds(f, tn::AbstractITensorNetwork, args...; kwargs...)
  ```

* Prime/tag family — apply the corresponding ITensors index-label operation to every
  index of the network (`abstractitensornetwork.jl`):
  ```julia
  prime(tn::AbstractITensorNetwork, args...; kwargs...)
  setprime(tn::AbstractITensorNetwork, args...; kwargs...)
  noprime(tn::AbstractITensorNetwork, args...; kwargs...)
  replaceprime(tn::AbstractITensorNetwork, args...; kwargs...)
  swapprime(tn::AbstractITensorNetwork, args...; kwargs...)
  addtags(tn::AbstractITensorNetwork, args...; kwargs...)
  removetags(tn::AbstractITensorNetwork, args...; kwargs...)
  replacetags(tn::AbstractITensorNetwork, args...; kwargs...)
  settags(tn::AbstractITensorNetwork, args...; kwargs...)
  swaptags(tn::AbstractITensorNetwork, args...; kwargs...)
  sim(tn::AbstractITensorNetwork, args...; kwargs...)
  ```

## IndsNetwork Type and Methods

* Build an `IndsNetwork` of site indices on `g` from a value `x` — site-type string,
  dimension, `Index`, or per-vertex dictionary (`sitetype.jl`):
  ```julia
  siteinds(x, g::AbstractGraph; kwargs...)
  ```

* Same, on a length-`nv` path graph (`sitetype.jl`):
  ```julia
  siteinds(x, nv::Int; kwargs...)
  ```

* Build an `IndsNetwork` by calling `f(v)` at each vertex (`sitetype.jl`):
  ```julia
  siteinds(f::Function, g::AbstractGraph; kwargs...)
  ```

## TEBD and Apply Algorithms

* Run TEBD given a set of Hamiltonian terms (`tebd.jl`):
  ```julia
  tebd(
        ℋ::Sum,
        ψ::AbstractITensorNetwork;
        β,
        Δβ,
        maxdim,
        cutoff,
        print_frequency = 10,
        ortho = false,
        kwargs...
    )
  ```

* Apply a set of gates to an ITensorNetwork (`apply.jl`):
  ```julia
  ITensors.apply(o::Union{NamedEdge, ITensor},ψ::AbstractITensorNetwork; kws...)
  ITensors.apply(o⃗::Union{Vector{NamedEdge}, Vector{ITensor}}, ψ::AbstractITensorNetwork; kws...)
  ITensors.apply(o⃗::Scaled,ψ::AbstractITensorNetwork; kws...)
  ITensors.apply(o⃗::Prod, ψ::AbstractITensorNetwork; kws...)
  ITensors.apply(o::Op, ψ::AbstractITensorNetwork; kws...)
  ```

## Visualization System

* Visualization of an ITensorNetwork via `ITensorVisualizationCore` (`abstractitensornetwork.jl`):
  ```julia
  visualize(tn::AbstractITensorNetwork, args...; kwargs...)
  ```
