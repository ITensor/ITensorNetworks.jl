# Experimental Methods

Methods which still need to be discussed, modified, or deprecated.

## ITensorNetwork Methods

* Combine (fuse) every link index of a tensor network, or a chosen set of edges, into
  a single index per edge using `combiner` tensors. (`abstractitensornetwork.jl`).
  Comment: it may be better to fold this into a more general interface `gauge_transform(f, tn::AbstractITensorNetwork)` that applies a gauge transformation $X\_e$ $X\_e^{-1}$ on each edge $e$.
  ```julia
  linkinds_combiners(tn::AbstractITensorNetwork; edges = edges(tn))
  combine_linkinds(tn::AbstractITensorNetwork, combiners)
  combine_linkinds(tn::AbstractITensorNetwork; edges = edges(tn))
  ```

* Construct an `ITensorNetwork` from an `IndsNetwork`. Initializes ITensors with `undef` storage on each vertex
  of the `IndsNetwork` with the corresponding indices (`itensornetwork.jl`):
  ```julia
  ITensorNetwork(eltype::Type, undef::UndefInitializer, is::IndsNetwork; kwargs...)
  ITensorNetwork(eltype::Type, is::IndsNetwork; kwargs...)
  ITensorNetwork(undef::UndefInitializer, is::IndsNetwork; kwargs...)
  ITensorNetwork(is::IndsNetwork; kwargs...)
  ```

* Extract the `IndsNetwork` of a tensor network — site indices per vertex and link
  indices per edge (`abstractitensornetwork.jl`):
  ```julia
  IndsNetwork(tn::AbstractITensorNetwork)
  ```

* Collect all site indices (per-vertex) of a network as an `IndsNetwork` (`abstractitensornetwork.jl`):
  ```julia
  siteinds(tn::AbstractITensorNetwork)
  ```

* Collect all link indices (per-edge) of a network as an `IndsNetwork` (`abstractitensornetwork.jl`):
  ```julia
  linkinds(tn::AbstractITensorNetwork)
  ```

* Rewrite every index of a network according to a structural mapping `IndsNetwork => IndsNetwork`
  (site indices per vertex, link indices per edge). The two `IndsNetwork`s must share the
  same underlying graph (`abstractitensornetwork.jl`):
  ```julia
  replaceinds(tn::AbstractITensorNetwork, is_is′::Pair{<:IndsNetwork, <:IndsNetwork})
  ```

* "Split" an edge index by applying a map to each copy of it on the adjacent ITensors.
  By default the `dst(edge)` copy is primed and the `src(edge)` copy is unchanged (`abstractitensornetwork.jl`):
  ```julia
  split_index(tn::AbstractITensorNetwork, edges_to_split; 
              src_ind_map::Function = identity,
              dst_ind_map::Function = prime)
  ```

* Collect all site indices (per-vertex) of a network as a flat vector (`abstractitensornetwork.jl`):
  ```julia
  flatten_siteinds(tn::AbstractITensorNetwork)
  ```

* Collect all link indices (per-edge) of a network as a flat vector (`abstractitensornetwork.jl`):
  ```julia
  flatten_linkinds(tn::AbstractITensorNetwork)
  ```


## TreeTensorNetwork Types

#### OpSum Constructors

* From an `OpSum`, using `opsum_to_ttn.jl` code:
  ```julia
  ttn(os::OpSum, sites::IndsNetwork; kws...)
  ```

* From `OpSum`, assuming path graph (`opsum_to_ttn.jl`):
  ```julia
  mpo(os::OpSum, external_inds::Vector; kws...)
  mpo(os::OpSum, s::IndsNetwork; kws...)
  ```

#### AbstractTreeTensorNetwork Type

* Required-to-implement abstract interface — `TreeTensorNetwork` provides all three (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  ITensorNetwork(tn::AbstractTTN)
  ortho_region(tn::AbstractTTN)
  set_ortho_region(tn::AbstractTTN, new_region)
  ```

* Underlying-graph type forwarded to `data_graph_type` (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  underlying_graph_type(G::Type{<:AbstractTTN})
  ```

* Gauge `tn` so its orthogonality center sits at `region`. `gauge` does the underlying
  tree-traversal QRs; `orthogonalize` is the user-facing wrapper, with `tree_orthogonalize`
  as an alias (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  gauge(alg::Algorithm, ttn::AbstractTTN, region::Vector; kwargs...)
  gauge(alg::Algorithm, ttn::AbstractTTN, region; kwargs...)
  orthogonalize(ttn::AbstractTTN, region; kwargs...)
  tree_orthogonalize(ttn::AbstractTTN, args...; kwargs...)
  ```

* Sweep-based truncation. The whole-TTN form orthogonalizes towards `src(e)` before
  each bond truncation; the edge form lifts `truncate` from the underlying `ITensorNetwork` (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  truncate(tn::AbstractTTN; root_vertex = GraphsExtensions.default_root_vertex(tn), kwargs...)
  truncate(tn::AbstractTTN, edge::AbstractEdge; kwargs...)
  ```

* Contract the whole tree into a single `ITensor` via a reverse post-order DFS sequence (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  contract(tn::AbstractTTN, root_vertex = GraphsExtensions.default_root_vertex(tn); kwargs...)
  ```

* Inner product `⟨x|y⟩`, matrix element `⟨y|A|x⟩`, and four-network form `⟨B|y|A|x⟩`,
  each contracted along a post-order DFS rooted at `root_vertex` (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  inner(x::AbstractTTN, y::AbstractTTN; root_vertex)
  inner(y::AbstractTTN, A::AbstractTTN, x::AbstractTTN; root_vertex)
  inner(B::AbstractTTN, y::AbstractTTN, A::AbstractTTN, x::AbstractTTN; root_vertex)
  ```

* Norm and `log(norm)` — fast paths when the gauge center is a single vertex (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  norm(tn::AbstractTTN)
  lognorm(tn::AbstractTTN)
  ```

* In-place and out-of-place normalization, distributing the norm across the gauge
  center (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  normalize!(tn::AbstractTTN)
  normalize(tn::AbstractTTN)
  ```

* Numerically-stable `log(⟨tn1|tn2⟩)` along a post-order DFS, accumulating per-step
  log-norms; `logdot` is an alias (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  loginner(tn1::AbstractTTN, tn2::AbstractTTN; root_vertex)
  logdot(tn1::AbstractTTN, tn2::AbstractTTN; kwargs...)
  ```

* Scalar arithmetic — multiplies/divides the gauge-center tensor by `α`, with `rmul!`
  as the in-place form (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  *(tn::AbstractTTN, α::Number)
  *(α::Number, tn::AbstractTTN)
  /(tn::AbstractTTN, α::Number)
  -(tn::AbstractTTN)
  rmul!(tn::AbstractTTN, α::Number)
  ```

* Add (or subtract) tree tensor networks by direct-summing bond indices. The result's
  bond dimension is the sum of the inputs'; the `Algorithm"directsum"` form is the
  current implementation. Use `truncate` afterward to compress (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  +(::Algorithm"directsum", tns::AbstractTTN...; root_vertex)
  +(tns::AbstractTTN...; alg = Algorithm"directsum"(), kwargs...)
  +(tn::AbstractTTN)
  -(tn1::AbstractTTN, tn2::AbstractTTN; kwargs...)
  add(tns::AbstractTTN...; kwargs...)
  add(tn1::AbstractTTN, tn2::AbstractTTN; kwargs...)
  ```

* Approximate equality via `norm(x - y) ≤ max(atol, rtol * max(norm(x), norm(y)))` (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  isapprox(x::AbstractTTN, y::AbstractTTN; atol, rtol)
  ```

* Local expectation values for a named operator at the given vertices (default all
  vertices), evaluated by successive orthogonalization (`treetensornetworks/abstracttreetensornetwork.jl`):
  ```julia
  expect(operator::String, state::AbstractTTN; vertices, root_vertex)
  ```

#### TreeTensorNetwork Type

* Get the underlying `ITensorNetwork` of a `TTN` (drops orthogonality metadata) (`treetensornetworks/treetensornetwork.jl`):
  ```julia
  ITensorNetwork(tn::TTN)
  ```

* Get the current orthogonality region — the set of vertices forming the gauge center (`treetensornetworks/treetensornetwork.jl`):
  ```julia
  ortho_region(tn::TTN)
  ```

* `AbstractITensorNetwork` interface, forwarded to the wrapped `ITensorNetwork` (`treetensornetworks/treetensornetwork.jl`):
  ```julia
  data_graph(tn::TTN)
  data_graph_type(G::Type{<:TTN})
  copy(tn::TTN)
  ```

* Low-level `ortho_region` update — only changes the metadata, performs no gauge
  transformations (use `orthogonalize` to actually move the gauge center) (`treetensornetworks/treetensornetwork.jl`):
  ```julia
  set_ortho_region(tn::TTN, ortho_region)
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

* Visualize an `IndsNetwork` by wrapping it in a default `ITensorNetwork` (`indsnetwork.jl`):
  ```julia
  visualize(is::IndsNetwork, args...; kwargs...)
  ```


## ProjTTN System

#### AbstractProjTTN

* Required-to-implement abstract interface — each concrete `AbstractProjTTN` subtype
  defines these (`treetensornetworks/projttns/abstractprojttn.jl`):
  ```julia
  environments(::AbstractProjTTN)
  operator(::AbstractProjTTN)
  pos(::AbstractProjTTN)
  underlying_graph(P::AbstractProjTTN)
  copy(::AbstractProjTTN)
  set_nsite(::AbstractProjTTN, nsite)
  shift_position(::AbstractProjTTN, pos)
  set_environments(p::AbstractProjTTN, environments)
  set_environment(p::AbstractProjTTN, edge, environment)
  make_environment!(P::AbstractProjTTN, psi, e)
  make_environment(P::AbstractProjTTN, psi, e)
  projected_operator_tensors(P::AbstractProjTTN)
  ```

* Position queries — whether the projection currently sits on an edge, the number
  and list of "sites" of the projection, and the corresponding incident / internal
  edges of the underlying graph (`treetensornetworks/projttns/abstractprojttn.jl`):
  ```julia
  edgetype(P::AbstractProjTTN)
  on_edge(P::AbstractProjTTN)
  nsite(P::AbstractProjTTN)
  sites(P::AbstractProjTTN)
  incident_edges(P::AbstractProjTTN)
  internal_edges(P::AbstractProjTTN)
  ```

* Look up a single environment tensor by edge (`treetensornetworks/projttns/abstractprojttn.jl`):
  ```julia
  environment(P::AbstractProjTTN, edge::AbstractEdge)
  environment(P::AbstractProjTTN, edge::Pair)
  ```

* Apply the projection to a vector — `contract(P, v)` does this in a literal way; 
   `product(P, v)` adds a `noprime` and an order check;
  `(P)(v)` is the callable form (`treetensornetworks/projttns/abstractprojttn.jl`):
  ```julia
  contract(P::AbstractProjTTN, v::ITensor)
  product(P::AbstractProjTTN, v::ITensor)
  (P::AbstractProjTTN)(v::ITensor)
  ```

* Eltype / vertextype / dim queries — `size` returns `(d, d)` from primed indices
  on environments and operator tensors (`treetensornetworks/projttns/abstractprojttn.jl`):
  ```julia
  eltype(P::AbstractProjTTN)
  vertextype(::Type{<:AbstractProjTTN{V}}) where {V}
  vertextype(p::AbstractProjTTN)
  size(P::AbstractProjTTN)
  ```

* Move the projection to a new `pos`: shifts position, drops now-internal-edge
  environments, and rebuilds the missing ones from `psi` (`treetensornetworks/projttns/abstractprojttn.jl`):
  ```julia
  position(P::AbstractProjTTN, psi::AbstractTTN, pos)
  ```

* Drop one or all internal-edge environments after a position change, and rebuild
  every incident-edge environment from `psi` (`treetensornetworks/projttns/abstractprojttn.jl`):
  ```julia
  invalidate_environment(P::AbstractProjTTN, e::AbstractEdge)
  invalidate_environments(P::AbstractProjTTN)
  make_environments(P::AbstractProjTTN, psi::AbstractTTN)
  ```

#### ProjTTN

* Construct a `ProjTTN` from an operator `TTN`. The two-argument form lets you specify
  position and pre-built environments; the one-argument form starts with empty
  environments and `pos = vertices(operator)` (`treetensornetworks/projttns/projttn.jl`):
  ```julia
  ProjTTN(pos, operator::TTN, environments::Dictionary)
  ProjTTN(operator::TTN)
  ```

* Field accessors and `copy` (`treetensornetworks/projttns/projttn.jl`):
  ```julia
  environments(p::ProjTTN)
  operator(p::ProjTTN)
  underlying_graph(P::ProjTTN)
  pos(P::ProjTTN)
  copy(P::ProjTTN)
  ```

* Position-management interface (`set_nsite` is a no-op for trees) (`treetensornetworks/projttns/projttn.jl`):
  ```julia
  set_nsite(P::ProjTTN, nsite)
  shift_position(P::ProjTTN, pos)
  ```

* Environment dictionary updates — out-of-place `set_environment` returns a copy
  with the bond's environment replaced; `set_environment!` is the in-place form (`treetensornetworks/projttns/projttn.jl`):
  ```julia
  set_environments(p::ProjTTN, environments)
  set_environment(p::ProjTTN, edge, env)
  set_environment!(p::ProjTTN, edge, env)
  ```

* Build the environment on edge `e` from `state` (`treetensornetworks/projttns/projttn.jl`):
  ```julia
  make_environment(P::ProjTTN, state::AbstractTTN, e::AbstractEdge)
  ```

* Assemble the ITensor list that defines the projection: incident-edge environments
  plus operator tensors at each site (`treetensornetworks/projttns/projttn.jl`):
  ```julia
  projected_operator_tensors(P::ProjTTN)
  ```

#### ProjTTNSum

* Construct a weighted sum of `AbstractProjTTN` terms, or a sum of `AbstractTTN`
  operators (which are wrapped via `ProjTTN.(operators)`). The two-argument form lets
  you specify per-term scalar factors (`treetensornetworks/projttns/projttnsum.jl`):
  ```julia
  ProjTTNSum(terms::Vector{<:AbstractProjTTN}, factors::Vector{<:Number})
  ProjTTNSum(operators::Vector{<:AbstractProjTTN})
  ProjTTNSum(operators::Vector{<:AbstractTTN})
  ```

* Field accessors and `copy` (`treetensornetworks/projttns/projttnsum.jl`):
  ```julia
  terms(P::ProjTTNSum)
  factors(P::ProjTTNSum)
  copy(P::ProjTTNSum)
  ```

* Position queries forwarded to the first term (`treetensornetworks/projttns/projttnsum.jl`):
  ```julia
  on_edge(P::ProjTTNSum)
  nsite(P::ProjTTNSum)
  underlying_graph(P::ProjTTNSum)
  length(P::ProjTTNSum)
  sites(P::ProjTTNSum)
  incident_edges(P::ProjTTNSum)
  internal_edges(P::ProjTTNSum)
  ```

* Update the position parameter on every term while preserving the factors (`treetensornetworks/projttns/projttnsum.jl`):
  ```julia
  set_nsite(Ps::ProjTTNSum, nsite)
  ```

* Apply the sum to a vector — `contract` builds `Σ fᵢ·contract(termᵢ, v)`,
  `product` adds the standard `noprime`, and `(P)(v)` is the callable form (`treetensornetworks/projttns/projttnsum.jl`):
  ```julia
  contract(P::ProjTTNSum, v::ITensor)
  product(P::ProjTTNSum, v::ITensor)
  (P::ProjTTNSum)(v::ITensor)
  ```

* Apply the sum without the bra side, used by ket-only projections like outer products (`treetensornetworks/projttns/projttnsum.jl`):
  ```julia
  contract_ket(P::ProjTTNSum, v::ITensor)
  ```

* Element type (promoted across terms) and `size` (taken from the first term) (`treetensornetworks/projttns/projttnsum.jl`):
  ```julia
  eltype(P::ProjTTNSum)
  size(P::ProjTTNSum)
  ```

* Move every term to a new `pos`, returning a fresh `ProjTTNSum` (`treetensornetworks/projttns/projttnsum.jl`):
  ```julia
  position(P::ProjTTNSum, psi::AbstractTTN, pos)
  ```

#### ProjOuterProdTTN

* Construct a `ProjOuterProdTTN` from an internal-state `TTN` and an operator `TTN`,
  starting at empty position with empty environments (`treetensornetworks/projttns/projouterprodttn.jl`):
  ```julia
  ProjOuterProdTTN(internal_state::AbstractTTN, operator::AbstractTTN)
  ```

* Field accessors and `copy` (`treetensornetworks/projttns/projouterprodttn.jl`):
  ```julia
  environments(p::ProjOuterProdTTN)
  operator(p::ProjOuterProdTTN)
  underlying_graph(p::ProjOuterProdTTN)
  pos(p::ProjOuterProdTTN)
  internal_state(p::ProjOuterProdTTN)
  copy(P::ProjOuterProdTTN)
  ```

* Position-management interface (`set_nsite` is a no-op) (`treetensornetworks/projttns/projouterprodttn.jl`):
  ```julia
  set_nsite(P::ProjOuterProdTTN, nsite)
  shift_position(P::ProjOuterProdTTN, pos)
  ```

* Environment dictionary updates — same pattern as `ProjTTN` (`treetensornetworks/projttns/projouterprodttn.jl`):
  ```julia
  set_environments(p::ProjOuterProdTTN, environments)
  set_environment(p::ProjOuterProdTTN, edge, env)
  set_environment!(p::ProjOuterProdTTN, edge, env)
  ```

* Build the environment on edge `e` from `state` — like `ProjTTN`'s version but
  uses the unprimed `internal_state` instead of priming `state` (`treetensornetworks/projttns/projouterprodttn.jl`):
  ```julia
  make_environment(P::ProjOuterProdTTN, state::AbstractTTN, e::AbstractEdge)
  ```

* Assemble the ITensor list that defines the projection — interleaves
  `internal_state`, operator, and environment tensors (`treetensornetworks/projttns/projouterprodttn.jl`):
  ```julia
  projected_operator_tensors(P::ProjOuterProdTTN)
  ```

* Apply the operator-with-internal-state combination to a vector. `contract_ket`
  performs the half-contraction with `internal_state`; `contract` returns
  `(dag(ket) · x) · ket` for outer-product evaluation (`treetensornetworks/projttns/projouterprodttn.jl`):
  ```julia
  contract_ket(P::ProjOuterProdTTN, v::ITensor)
  contract(P::ProjOuterProdTTN, x::ITensor)
  ```
