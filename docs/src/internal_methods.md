# Internal Methods

Developer-focused methods for internal use by other parts of ITensorNetworks.

## Required `AbstractITensorNetwork` Interface

Concrete subtypes of `AbstractITensorNetwork` must implement these — they are stubs
that throw `not_implemented()` on `AbstractITensorNetwork`.

* Underlying `DataGraph` and its type (`abstractitensornetwork.jl`):
  ```julia
  data_graph(tn::AbstractITensorNetwork)
  data_graph_type(::Type{<:AbstractITensorNetwork})
  copy(tn::AbstractITensorNetwork)
  ```

* Internal "private" inner constructor of `ITensorNetwork` that wraps a pre-built `DataGraph` (`itensornetwork.jl`):
  ```julia
  _ITensorNetwork(data_graph::DataGraph)
  ```

## ITensorNetwork-Building Helpers

* Build an `ITensorNetwork` from any vertex-indexed collection of `ITensor`s, inferring
  edges from shared indices (`itensornetwork.jl`):
  ```julia
  itensors_to_itensornetwork(ts)
  ```

* Build a per-vertex tensor from a "value" (function, array, string, `Op`, etc.) and
  the vertex's site/link indices. Used by the value-based `ITensorNetwork` constructors (`itensornetwork.jl`):
  ```julia
  generic_state(f, inds::NamedTuple)
  generic_state(f, inds::Vector)
  ```

* Convert a "value" (function, type, dict, dictionary, array, scalar) into a callable
  `v -> value_for_v` for use in vertex-wise tensor construction (`itensornetwork.jl`):
  ```julia
  to_callable(value)
  ```

* Insert default link indices on every edge of an `IndsNetwork` that doesn't have one (`abstractitensornetwork.jl`):
  ```julia
  insert_linkinds(is::IndsNetwork; link_space)
  ```

## Graph / Data-Graph Plumbing

* Underlying graph and graph-type accessors (`abstractitensornetwork.jl`):
  ```julia
  underlying_graph(tn::AbstractITensorNetwork)
  underlying_graph_type(G::Type{<:AbstractITensorNetwork})
  ```

* Vertex- and edge-data accessors, forwarded to `DataGraphs` (`abstractitensornetwork.jl`):
  ```julia
  vertex_data(graph::AbstractITensorNetwork, args...)
  edge_data(graph::AbstractITensorNetwork, args...)
  vertex_positions(tn::AbstractITensorNetwork)
  ordered_vertices(tn::AbstractITensorNetwork)
  ```

* Convert an `AbstractITensorNetwork` to its underlying simple/named graph, dropping tensors (`abstractitensornetwork.jl`):
  ```julia
  Graph(tn::AbstractITensorNetwork)
  NamedGraph(tn::AbstractITensorNetwork)
  ```

* Edge weights, uniform unit weights used for graph algorithms (`abstractitensornetwork.jl`):
  ```julia
  weights(graph::AbstractITensorNetwork)
  ```

## In-Place / Graph-Preserving Mutation

* Set a vertex's tensor without re-deriving graph edges from indices (`abstractitensornetwork.jl`):
  ```julia
  setindex_preserve_graph!(tn::AbstractITensorNetwork, value, vertex)
  ```

* Macro that wraps a `tn[v] = ...` (or block of such assignments) so the graph is
  preserved instead of being recomputed from index sharing (`abstractitensornetwork.jl`):
  ```julia
  @preserve_graph expr
  ```

* Apply a function to each vertex tensor, returning a new (or in-place) network.
  The `_preserve_graph` variants do not re-derive edges (`abstractitensornetwork.jl`):
  ```julia
  map_vertex_data(f, tn::AbstractITensorNetwork)
  map_vertex_data_preserve_graph(f, tn::AbstractITensorNetwork)
  map_vertices_preserve_graph!(f, tn::AbstractITensorNetwork; vertices = vertices(tn))
  ```

## Index / Network Queries

* Whether the tensors at the endpoints of `edge` share any index (`abstractitensornetwork.jl`):
  ```julia
  hascommoninds(tn::AbstractITensorNetwork, edge::AbstractEdge)
  hascommoninds(tn::AbstractITensorNetwork, edge::Pair)
  ```

* Indices common to two networks, matched across all tensor pairs (`abstractitensornetwork.jl`):
  ```julia
  commoninds(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
  ```

* Whether an edge carries more than one link index, and a curried predicate version (`abstractitensornetwork.jl`):
  ```julia
  is_multi_edge(tn::AbstractITensorNetwork, e)
  is_multi_edge(tn::AbstractITensorNetwork)
  ```

* Whether any tensor in the network carries QN block structure (`abstractitensornetwork.jl`):
  ```julia
  hasqns(tn::AbstractITensorNetwork)
  ```

* The vertices whose tensors share an index with an external `ITensor` `T` (`abstractitensornetwork.jl`):
  ```julia
  neighbor_vertices(ψ::AbstractITensorNetwork, T::ITensor)
  ```

## Element-Type / Adapt System

* Element type and storage type of the tensors in the network (`abstractitensornetwork.jl`):
  ```julia
  datatype(tn::AbstractITensorNetwork)
  promote_indtypeof(tn::AbstractITensorNetwork)
  ```

* GPU/Adapt support — apply `adapt(to, ·)` to every tensor (`abstractitensornetwork.jl`):
  ```julia
  adapt_structure(to, tn::AbstractITensorNetwork)
  ```

## Local Operations on ITensorNetworks

* Gauge a single edge of an ITensorNetwork, moving the orthogonality/gauge across the bond (`abstractitensornetwork.jl`):
  ```julia
  gauge_edge(tn::AbstractITensorNetwork, edge::AbstractEdge)
  ```

* Internal worker for edge truncation (SVD across a bond, drop small singular values).
  Called by `truncate(tn, edge)` (`abstractitensornetwork.jl`):
  ```julia
  _truncate_edge(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
  ```

## Global Operations on ITensorNetworks

* Form-network helpers — build the bilinear/quadratic form network for inner products (`abstractitensornetwork.jl`):
  ```julia
  inner_network(x::AbstractITensorNetwork, y::AbstractITensorNetwork; kwargs...)
  norm_sqr_network(ψ::AbstractITensorNetwork)
  ```

* Apply a sequence of edge gauges to walk the gauge center along the given edges (`abstractitensornetwork.jl`):
  ```julia
  gauge_walk(alg::Algorithm, tn::AbstractITensorNetwork, edges::Vector{<:AbstractEdge}; kws...)
  gauge_walk(alg::Algorithm, tn::AbstractITensorNetwork, edge::Pair; kws...)
  gauge_walk(alg::Algorithm, tn::AbstractITensorNetwork, edges::Vector{<:Pair}; kws...)
  ```

* Gauge an ITensorNetwork towards a region, treating the network as a tree spanned by a
  spanning tree (`abstractitensornetwork.jl`):
  ```julia
  tree_gauge(alg::Algorithm, ψ::AbstractITensorNetwork, cur_region::Vector, new_region::Vector; kws...)
  tree_gauge(alg::Algorithm, ψ::AbstractITensorNetwork, region)
  tree_gauge(alg::Algorithm, ψ::AbstractITensorNetwork, region::Vector)
  ```

* Orthogonalize an ITensorNetwork towards a region along the spanning tree (`abstractitensornetwork.jl`):
  ```julia
  tree_orthogonalize(ψ::AbstractITensorNetwork, cur_region, new_region; kwargs...)
  tree_orthogonalize(ψ::AbstractITensorNetwork, region; kwargs...)
  ```

## Apply System

* BP algorithms used by `apply` (`apply.jl`):
  ```julia
  full_update_bp(o::Union{NamedEdge, ITensor},ψ,v; kws...)
  simple_update_bp_full(o::Union{NamedEdge, ITensor}, ψ, v; kws...)
  simple_update_bp(o::Union{NamedEdge, ITensor}, ψ, v; kws...)
  ```

* Helper functions for `full_update_bp` (`apply.jl`):
  ```julia
  optimise_p_q(p::ITensor,q::ITensor,envs::Vector{ITensor},o::ITensor; kws...)
  # Cost function for optimise_p_q:
  fidelity(envs::Vector{ITensor},p_cur::ITensor,q_cur::ITensor,p_prev::ITensor,q_prev::ITensor,gate::ITensor)
  ```

## Index and Siteinds Helper Methods

* Create appropriate Index tag for vertices (`indextags.jl`):
  ```julia
  vertex_tag(v)
  ```
  Used by `edge_tag` below and in `sitetype.jl`.

* Create appropriate Index tag for edges (`indextags.jl`):
  ```julia
  edge_tag(e)
  ```
  Used in `abstractindsnetwork.jl`, `apply.jl`, `abstractitensornetwork.jl`, `treetensornetworks/opsum_to_ttn/opsum_to_ttn.jl`, `treetensornetworks/treetensornetwork.jl`.

* Create an Index from a vertex and a space (`indextags.jl`):
  ```julia
  vertex_index(v, vertex_space)
  ```

* Create an Index from an edge and a space (`indextags.jl`):
  ```julia
  edge_index(e, edge_space)
  ```

## IndsNetwork Type and Methods

* Build a single site `Index` for vertex `v`, tagged with `vertex_tag(v)` (`sitetype.jl`):
  ```julia
  ITensors.siteind(sitetype::String, v::Tuple; kwargs...)
  ITensors.siteind(d::Integer, v; addtags = "", kwargs...)
  ```

* Convert a "value" into a callable `v -> value_for_v`, used by the value-based
  `siteinds` form (`sitetype.jl`):
  ```julia
  to_siteinds_callable(x)
  to_siteinds_callable(x::AbstractDictionary)
  ```

* Wrap the per-vertex result of `siteinds(f, g)` into a `Vector{Index}` for that
  vertex, building a fresh `Index` when needed (`sitetype.jl`):
  ```julia
  to_siteind(x, vertex; kwargs...)
  to_siteind(x::Index, vertex; kwargs...)
  ```

