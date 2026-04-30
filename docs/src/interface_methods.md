# Interface Methods

Recommended methods for building applications on top of ITensorNetworks.

## ITensorNetwork Constructors

These ITensorNetwork constructor interfaces are foundational to other constructors:

* Copy constructor (`itensornetwork.jl`):
  ```julia
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


## Analyzing ITensorNetworks


* Tags on the link index (or indices) associated with `edge` (`abstractitensornetwork.jl`):
  ```julia
  tags(tn::AbstractITensorNetwork, edge)
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

* Scale tensors at chosen vertices by per-vertex weights, either out-of-place or in-place (`abstractitensornetwork.jl`). Comment: should probably be renamed to `scale_tensors`.
  ```julia
  scale(tn::AbstractITensorNetwork, vertices_weights::Dictionary; kwargs...)
  scale(weight_function::Function, tn; kwargs...)
  scale!(tn::AbstractITensorNetwork, vertices_weights::Dictionary)
  scale!(weight_function::Function, tn::AbstractITensorNetwork; kwargs...)
  ```

* Tensor product (disjoint union) of two ITensorNetworks (`abstractitensornetwork.jl`):
  ```julia
  ⊗(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
  union(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; kwargs...)
  ```

* Contract every tensor in the network into a single `ITensor`. Default `alg = "exact"`
  contracts via a contraction sequence (built from the network if not given) (`contract.jl`):
  ```julia
  contract(tn::AbstractITensorNetwork; alg, kwargs...)
  contract(alg::Algorithm"exact", tn::AbstractITensorNetwork; sequence, contraction_sequence_kwargs, kwargs...)
  ```

* Scalar value of a fully-contracted network. The `Algorithm"exact"` form contracts and
  unwraps; the generic `Algorithm` form goes through `logscalar`/`exp` for stability (`contract.jl`):
  ```julia
  scalar(tn::AbstractITensorNetwork; alg, kwargs...)
  scalar(alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...)
  scalar(alg::Algorithm, tn::AbstractITensorNetwork; kwargs...)
  ```

* `log` of the network scalar. The `Algorithm"exact"` form contracts and takes a log
  (promoting to complex when negative); the generic `Algorithm` form goes through a
  cache (e.g. BP) using `cache!` / `update_cache` (`contract.jl`):
  ```julia
  logscalar(tn::AbstractITensorNetwork; alg, kwargs...)
  logscalar(alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...)
  logscalar(alg::Algorithm, tn::AbstractITensorNetwork; cache!, update_cache, kwargs...)
  ```

* Obtain contraction sequence for a tensor network (`contraction_sequences.jl`).
  Can offer different backends through package extensions.
  ```julia
  contraction_sequence(tn::ITensorList; alg = "optimal", kwargs...)
  contraction_sequence(alg::Algorithm, tn::ITensorList)
  contraction_sequence(tn::AbstractITensorNetwork; kwargs...)
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

* Tensors making up the environment of `vertices` in `tn`. Default `alg = "bp"` (`environment.jl`):
  ```julia
  environment(tn::AbstractITensorNetwork, vertices::Vector; alg, kwargs...)
  ```

* Algorithm-specialized dispatches: `"exact"` returns the single ITensor obtained by
  contracting all other vertices; the generic `Algorithm` form partitions the network
  (or accepts a `PartitionedGraph`) and pulls the environment from a BP-style cache (`environment.jl`):
  ```julia
  environment(::Algorithm"exact", tn::AbstractITensorNetwork, verts::Vector; kwargs...)
  environment(alg::Algorithm, tn::AbstractITensorNetwork, vertices::Vector; partitioned_vertices, kwargs...)
  environment(alg::Algorithm, ptn::PartitionedGraph, vertices::Vector; cache!, update_cache, kwargs...)
  ```

## Index Manipulation

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


## Solvers System

* Find the lowest eigenvalue / eigenvector of `operator` via a DMRG-style sweep on a
  `TreeTensorNetwork`. `dmrg` is an alias for `eigsolve` (`solvers/eigsolve.jl`):
  ```julia
  eigsolve(operator, init_state; nsweeps, nsites = 1, factorize_kwargs, sweep_callback, sweep_kwargs...)
  dmrg(operator, init_state; kwargs...)
  ```

* Apply `exp(exponents[i])·operator` to `init_state` along a sequence of exponent
  values, using a sweep-based local solver (Runge–Kutta by default). The
  pre-built-problem form lets a caller drive `applyexp` from a custom `AbstractProblem` (`solvers/applyexp.jl`):
  ```julia
  applyexp(operator, exponents, init_state; sweep_callback, order, nsites, sweep_kwargs...)
  applyexp(init_prob::AbstractProblem, exponents; sweep_callback, order, nsites, sweep_kwargs...)
  ```

* Time-evolve `init_state` under `operator` using TDVP — wraps `applyexp` with
  `exponents = -im .* time_points`. Supports real and complex `time_points` (`solvers/applyexp.jl`):
  ```julia
  time_evolve(operator, time_points, init_state; sweep_kwargs...)
  ```

