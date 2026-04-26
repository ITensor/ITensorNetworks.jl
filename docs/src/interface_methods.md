# Interface Methods

## Files to Review

- [X] itensornetwork.jl
- [X] abstractitensornetwork.jl
- [X] apply.jl. Applying gates to ITensorNetworks.
- [X] tebd.jl

- [X] inner.jl
- [X] expect.jl
- [X] normalize.jl

- [X] edge\_sequences.jl
- [X] caches/abstractbeliefpropagationcache.jl
- [X] caches/beliefpropagationcache.jl

- [X] indextags.jl
- [X] sitetype.jl
- [X] abstractindsnetwork.jl
- [X] indsnetwork.jl

- [X] graphs.jl
- [X] contract.jl
- [X] contraction\_sequences.jl

- [ ] partitioneditensornetwork.jl
- [ ] specialitensornetworks.jl

- [ ] environment.jl

- [ ] opsum.jl

- [X] update\_observer.jl
- [X] utils.jl
- [X] visualize.jl


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

#### Site Index Helpers

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

#### AbstractIndsNetwork

* Required-to-implement abstract interface — concrete subtypes must define
  `data_graph`; `is_directed` defaults to `false` and may be overloaded (`abstractindsnetwork.jl`):
  ```julia
  data_graph(graph::AbstractIndsNetwork)
  is_directed(::Type{<:AbstractIndsNetwork})
  ```

* Vertex/edge data forwarded from the underlying `DataGraph`, plus the eltype
  declaration `Vector{I}` (`abstractindsnetwork.jl`):
  ```julia
  vertex_data(graph::AbstractIndsNetwork, args...)
  edge_data(graph::AbstractIndsNetwork, args...)
  edge_data_eltype(::Type{<:AbstractIndsNetwork{V, I}}) where {V, I}
  ```

* Indices "unique" to one side of an edge — for `edge`, returns the indices on
  `src(edge)` together with all of its other incident-edge link indices (`abstractindsnetwork.jl`):
  ```julia
  uniqueinds(is::AbstractIndsNetwork, edge::AbstractEdge)
  uniqueinds(is::AbstractIndsNetwork, edge::Pair)
  ```

* Merge two `AbstractIndsNetwork`s, returning an `IndsNetwork` over the merged graph (`abstractindsnetwork.jl`):
  ```julia
  union(is1::AbstractIndsNetwork, is2::AbstractIndsNetwork; kwargs...)
  ```

* Rename every vertex `v` to `f(v)` (`abstractindsnetwork.jl`):
  ```julia
  rename_vertices(f::Function, tn::AbstractIndsNetwork)
  ```

* Promoted index type across all site and link indices in the network (`abstractindsnetwork.jl`):
  ```julia
  promote_indtypeof(is::AbstractIndsNetwork)
  ```

* Build an `IndsNetwork` whose site/link indices at each vertex/edge are the union
  of the corresponding indices from each input network (graphs must match) (`abstractindsnetwork.jl`):
  ```julia
  union_all_inds(is_in::AbstractIndsNetwork...)
  ```

* Insert a default link index on every edge of `indsnetwork` that doesn't already
  have one — `link_space` controls the default bond dimension (`abstractindsnetwork.jl`):
  ```julia
  insert_linkinds(indsnetwork::AbstractIndsNetwork, edges = edges(indsnetwork); link_space = trivial_space(indsnetwork))
  ```

#### IndsNetwork

* Type-parameter accessors and graph-type metadata (`indsnetwork.jl`):
  ```julia
  indtype(inds_network::IndsNetwork)
  indtype(::Type{<:IndsNetwork{V, I}}) where {V, I}
  data_graph(is::IndsNetwork)
  underlying_graph(is::IndsNetwork)
  vertextype(::Type{<:IndsNetwork{V}}) where {V}
  underlying_graph_type(G::Type{<:IndsNetwork})
  is_directed(::Type{<:IndsNetwork})
  ```

* Construct an `IndsNetwork` from a pre-built `DataGraph` (`indsnetwork.jl`):
  ```julia
  IndsNetwork{V, I}(data_graph::DataGraph)
  IndsNetwork{V}(data_graph::DataGraph)
  IndsNetwork(data_graph::DataGraph)
  ```

* Construct from an underlying graph plus link- and site-space specs (positional
  or as `link_space` / `site_space` kwargs). Each spec may be an integer, a `Vector{Int}`,
  an `Index`, a `Vector{<:Index}`, or a per-edge / per-vertex `Dictionary` of any of
  those. `nothing` leaves it empty (`indsnetwork.jl`):
  ```julia
  IndsNetwork{V, I}(g::AbstractNamedGraph, link_space, site_space)
  IndsNetwork{V, I}(g::AbstractSimpleGraph, link_space, site_space)
  IndsNetwork{V}(g, link_space, site_space)
  IndsNetwork(g, link_space, site_space)
  IndsNetwork{V, I}(g; link_space, site_space)
  IndsNetwork{V}(g; link_space, site_space)
  IndsNetwork(g; kwargs...)
  ```

* Core constructor — takes pre-built `Dictionary` link- and site-space maps and
  populates the underlying `DataGraph` directly (`indsnetwork.jl`):
  ```julia
  IndsNetwork{V, I}(g::AbstractNamedGraph, link_space::Dictionary, site_space::Dictionary)
  IndsNetwork{V, I}(g::AbstractSimpleGraph, link_space::Dictionary, site_space::Dictionary)
  ```

* Build an `IndsNetwork` on a path graph from a vector of external (site) indices
  per vertex, or one index per vertex (`indsnetwork.jl`):
  ```julia
  path_indsnetwork(external_inds::Vector{<:Vector{<:Index}})
  path_indsnetwork(external_inds::Vector{<:Index})
  ```

* Normalize a user-supplied link-space spec into a `Dictionary{edgetype, Vector{I}}`,
  building fresh edge-tagged `Index` objects from raw integer dimensions when needed.
  Accepts an integer, a `Vector{Int}`, a `Dictionary` of integers / vectors of integers /
  `Index` / `Vector{<:Index}`, or `nothing` (`indsnetwork.jl`):
  ```julia
  link_space_map(V::Type, I::Type{<:Index}, g, link_space)
  ```

* Same for site spaces, normalizing into `Dictionary{V, Vector{I}}` (`indsnetwork.jl`):
  ```julia
  site_space_map(V::Type, I::Type{<:Index}, g, site_space)
  ```

* Copy an `IndsNetwork` (deep-copies the underlying `DataGraph`) (`indsnetwork.jl`):
  ```julia
  copy(is::IndsNetwork)
  ```

* Apply an index-label transformation `f` to every site index (`sites` kwarg) and/or
  link index (`links` kwarg) of the network (`indsnetwork.jl`):
  ```julia
  map_inds(f, is::IndsNetwork, args...; sites = nothing, links = nothing, kwargs...)
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

* Visualize an `IndsNetwork` by wrapping it in a default `ITensorNetwork` (`indsnetwork.jl`):
  ```julia
  visualize(is::IndsNetwork, args...; kwargs...)
  ```

## Caches for BP

#### Edge Sequence Helpers

* Build an edge traversal sequence over `g`, dispatched by `alg` (default `"forest_cover"`).
  Directed graphs are handled by undirecting first (`edge_sequences.jl`):
  ```julia
  edge_sequence(g; alg, kwargs...)
  edge_sequence(alg::Algorithm, g; kwargs...)
  ```

* Tree-traversal sequence: cover `g` with a forest, then for each tree push a post-order
  DFS sweep followed by its reverse — gives a back-and-forth sequence covering every edge (`edge_sequences.jl`):
  ```julia
  edge_sequence(::Algorithm"forest_cover", g; root_vertex = GraphsExtensions.default_root_vertex)
  ```

* Parallel sequence: each edge (and its reverse) as its own one-edge group, suitable
  for parallel BP updates (`edge_sequences.jl`):
  ```julia
  edge_sequence(::Algorithm"parallel", g)
  ```

#### AbstractBeliefPropagationCache

* Required-to-implement abstract interface — concrete subtypes must define these
  (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  # How many of these are user-facing versus internal?
  setindex!(bpc::AbstractBeliefPropagationCache, factor::ITensor, vertex)
  partitioned_tensornetwork(bpc::AbstractBeliefPropagationCache)
  messages(bpc::AbstractBeliefPropagationCache)
  copy(bpc::AbstractBeliefPropagationCache)
  partitions(bpc::AbstractBeliefPropagationCache)
  quotientedges(bpc::AbstractBeliefPropagationCache)
  partitioned_vertices(bpc::AbstractBeliefPropagationCache)
  environment(bpc::AbstractBeliefPropagationCache, verts::Vector; kwargs...)
  region_scalar(bpc::AbstractBeliefPropagationCache, pv::QuotientVertex; kwargs...)
  region_scalar(bpc::AbstractBeliefPropagationCache, pe::QuotientEdge; kwargs...)
  ```

* Forward type plumbing to the underlying tensor network and access the unpartitioned
  network behind a cache (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  similar_type(bpc::AbstractBeliefPropagationCache)
  data_graph_type(bpc::AbstractBeliefPropagationCache)
  data_graph(bpc::AbstractBeliefPropagationCache)
  tensornetwork(bpc::AbstractBeliefPropagationCache)
  scalartype(bpc::AbstractBeliefPropagationCache)
  ```

* Partition-graph queries forwarded to the underlying `PartitionedGraph` (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  vertices(bpc::AbstractBeliefPropagationCache)
  quotient_graph(bpc::AbstractBeliefPropagationCache)
  quotientedge(bpc::AbstractBeliefPropagationCache, edge::AbstractEdge)
  quotientvertices(bpc::AbstractBeliefPropagationCache)
  quotientvertices(bpc::AbstractBeliefPropagationCache, vs)
  boundary_quotientedges(bpc::AbstractBeliefPropagationCache, quotientvertices; kwargs...)
  boundary_quotientedges(bpc::AbstractBeliefPropagationCache, quotientvertex::QuotientVertex; kwargs...)
  linkinds(bpc::AbstractBeliefPropagationCache, pe::QuotientEdge)
  ```

* Vertex tensors (factors) at given vertices, or for all vertices in the given
  partitions (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  factors(bpc::AbstractBeliefPropagationCache, verts::Vector)
  factors(bpc::AbstractBeliefPropagationCache, partition_verts::Vector{<:QuotientVertex})
  factors(bpc::AbstractBeliefPropagationCache, partition_vertex::QuotientVertex)
  ```

* Out-of-place factor updates — replace one or many vertex tensors, returning a new cache (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  update_factor(bpc, vertex, factor)
  update_factors(bpc::AbstractBeliefPropagationCache, factors)
  ```

* Apply a function to every (or a chosen subset of) factor tensors (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  map_factors(f, bpc::AbstractBeliefPropagationCache, vs = vertices(bpc))
  ```

* Read messages from the cache: a single message (with default fallback when missing),
  or a vector of messages on the given edges (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  message(bpc::AbstractBeliefPropagationCache, edge::QuotientEdge; kwargs...)
  messages(bpc::AbstractBeliefPropagationCache, edges; kwargs...)
  ```

* Set messages — in-place (`!`) variants mutate the cache, the non-`!` variants return
  a new cache (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  set_message!(bpc::AbstractBeliefPropagationCache, pe::QuotientEdge, message)
  set_messages!(bpc::AbstractBeliefPropagationCache, quotientedges_messages)
  set_message(bpc::AbstractBeliefPropagationCache, pe::QuotientEdge, message)
  set_messages(bpc::AbstractBeliefPropagationCache, quotientedges_messages)
  ```

* Delete messages on chosen edges (or all edges); same in-place vs out-of-place
  convention as `set_message[s]` (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  delete_message!(bpc::AbstractBeliefPropagationCache, pe::QuotientEdge)
  delete_messages!(bpc::AbstractBeliefPropagationCache, pes::Vector{<:QuotientEdge} = keys(messages(bpc)))
  delete_message(bpc::AbstractBeliefPropagationCache, pe::QuotientEdge)
  delete_messages(bpc::AbstractBeliefPropagationCache, pes::Vector{<:QuotientEdge} = keys(messages(bpc)))
  ```

* Apply a function to every (or a chosen subset of) message tensors (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  map_messages(f, bpc::AbstractBeliefPropagationCache, pes = collect(keys(messages(bpc))))
  ```

* Collect the messages flowing into a partition vertex (or set of partition vertices),
  optionally ignoring some edges (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  incoming_messages(bpc::AbstractBeliefPropagationCache, partition_vertices::Vector{<:QuotientVertex}; ignore_edges = ())
  incoming_messages(bpc::AbstractBeliefPropagationCache, partition_vertex::QuotientVertex; kwargs...)
  ```

* Convergence proxy: `1 - |⟨a|b⟩|²` between two normalized contracted messages (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  message_diff(message_a::Vector{ITensor}, message_b::Vector{ITensor})
  ```

* Adapt support — propagate `adapt(to, ·)` over messages, factors, or both (used for
  GPU/eltype migration) (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  adapt_messages(to, bpc::AbstractBeliefPropagationCache, args...)
  adapt_factors(to, bpc::AbstractBeliefPropagationCache, args...)
  adapt_structure(to, bpc::AbstractBeliefPropagationCache)
  ```

* Region scalars per partition vertex / partition edge, and the convenient pair of
  vectors used by `logscalar` (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  vertex_scalars(bpc::AbstractBeliefPropagationCache, pvs = partitions(bpc); kwargs...)
  edge_scalars(bpc::AbstractBeliefPropagationCache, pes = quotientedges(bpc); kwargs...)
  scalar_factors_quotient(bpc::AbstractBeliefPropagationCache)
  ```

* The (log of the) BP estimate of the network scalar `⟨tn⟩` from current messages (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  logscalar(bpc::AbstractBeliefPropagationCache)
  scalar(bpc::AbstractBeliefPropagationCache)
  ```

* Compute an updated message on `edge` — either via straight contraction (with optional
  normalization), via an `adapt`-then-update wrapper, or via the user-facing dispatcher
  that fills in defaults (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  updated_message(alg::Algorithm"contract", bpc::AbstractBeliefPropagationCache, edge::QuotientEdge)
  updated_message(alg::Algorithm"adapt_update", bpc::AbstractBeliefPropagationCache, edge::QuotientEdge)
  updated_message(bpc::AbstractBeliefPropagationCache, edge::QuotientEdge; alg, kwargs...)
  ```

* Compute the updated message and write it back into a copied cache (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  update_message(message_update_alg::Algorithm, bpc::AbstractBeliefPropagationCache, edge::QuotientEdge)
  ```

* One BP iteration — sequential over a list of edges, or in parallel groups of edges,
  with an optional accumulator for per-iteration message diffs (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  update_iteration(alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache, edges::Vector; update_diff!)
  update_iteration(alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache, edge_groups::Vector{<:Vector{<:QuotientEdge}}; update_diff!)
  ```

* Drive BP to convergence — either the algorithm-specialized core or the user-facing
  dispatcher that fills in `maxiter`, `tol`, and the edge sequence (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  update(alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache)
  update(bpc::AbstractBeliefPropagationCache; alg, kwargs...)
  ```

* Rescale messages on a single bond, on the given bonds, or on every bond, so that
  `region_scalar` of the bond is 1 (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  rescale_messages(bp_cache::AbstractBeliefPropagationCache, quotientedge::QuotientEdge)
  rescale_messages(bp_cache::AbstractBeliefPropagationCache, pes)
  rescale_messages(bp_cache::AbstractBeliefPropagationCache)
  ```

* Rescale the vertex tensors inside one partition, a list of partitions, or every
  partition, so that the partition's region scalar is 1 (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  rescale_partition(bpc::AbstractBeliefPropagationCache, partition, args...; kwargs...)
  rescale_partitions(bpc::AbstractBeliefPropagationCache, partitions::Vector; verts)
  rescale_partitions(bpc::AbstractBeliefPropagationCache, args...; kwargs...)
  ```

* Convenience — rescale messages and then partitions in one call (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  rescale(bpc::AbstractBeliefPropagationCache, args...; kwargs...)
  ```

#### BeliefPropagationCache

* Construct a `BeliefPropagationCache` from an
  `AbstractITensorNetwork` plus a partitioning, or just a network (using a default
  partition) (`caches/beliefpropagationcache.jl`):
  ```julia
  BeliefPropagationCache(tn::AbstractITensorNetwork, partitioned_vertices; kwargs...)
  BeliefPropagationCache(tn::AbstractITensorNetwork; partitioned_vertices, kwargs...)
  # Deprecate this version?
  BeliefPropagationCache(ptn::PartitionedGraph; messages = default_messages(ptn))
  ```

* Given a tensor network, returns a `BeliefPropagationCache` (`caches/beliefpropagationcache.jl`):
  ```julia
  cache(alg::Algorithm"bp", tn; kwargs...)
  ```

* Copy cache and access messages (`caches/beliefpropagationcache.jl`):
  ```julia
  copy(bp_cache::BeliefPropagationCache)
  messages(bp_cache::BeliefPropagationCache)
  setindex!(bpc::BeliefPropagationCache, factor::ITensor, vertex)
  ```

* Partition-graph related queries — list partitions, quotient edges between partitions, and
  the vertex-to-partition mapping (`caches/beliefpropagationcache.jl`):
  ```julia
  partitioned_tensornetwork(bp_cache::BeliefPropagationCache)
  partitions(bpc::BeliefPropagationCache)
  quotientedges(bpc::BeliefPropagationCache)
  partitioned_vertices(bpc::BeliefPropagationCache)
  ```

* Environment around `verts`: incoming BP messages plus the in-partition tensors not in
  `verts`, suitable for local contractions (`caches/beliefpropagationcache.jl`):
  ```julia
  environment(bpc::BeliefPropagationCache, verts::Vector; kwargs...)
  ```

* Scalar associated with a region — for a partition vertex it's the local state contracted
  with incoming messages; for a partition edge it's the inner product of the two messages
  on that bond (`caches/beliefpropagationcache.jl`):
  ```julia
  region_scalar(bp_cache::BeliefPropagationCache, pv::QuotientVertex)
  region_scalar(bp_cache::BeliefPropagationCache, pe::QuotientEdge)
  ```

* Return a copy of the cache with messages on `pes` rescaled to unit norm and
  symmetrized so the bond region scalar is 1 (`caches/beliefpropagationcache.jl`):
  ```julia
  rescale_messages(bp_cache::BeliefPropagationCache, pes)
  ```
