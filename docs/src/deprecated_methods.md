# Deprecated Methods

Suggestions of methods which could be deleted.

## ITensorNetwork Methods

#### ITensorNetwork Constructors

* From a named graph, forwards to construction from `IndsNetwork` (`itensornetwork.jl`):
  ```julia
  ITensorNetwork{V}(g::NamedGraph)
  ITensorNetwork(eltype::Type, undef::UndefInitializer, graph::AbstractNamedGraph; kws...)
  ITensorNetwork(f, graph::AbstractNamedGraph; kwargs...)
  ITensorNetwork(graph::AbstractNamedGraph; kwargs...)
  ```

* From a simple graph, forwards to construction from `IndsNetwork` (`itensornetwork.jl`):
  ```julia
  ITensorNetwork(eltype::Type, undef::UndefInitializer, graph::AbstractSimpleGraph; kws...)
  ITensorNetwork(f, graph::AbstractSimpleGraph; kwargs...)
  ITensorNetwork(graph::AbstractSimpleGraph; kwargs...)
  ```

* From a function over vertices or from a "value" (e.g. a string like `"Up"`,
  an `Op`, an array, or a per-vertex dict/array) that is converted to a callable and used
  to initialize each vertex tensor (`itensornetwork.jl`):
  ```julia
  ITensorNetwork(value, is::IndsNetwork; kwargs...)
  ITensorNetwork(elt::Type, f, is::IndsNetwork; link_space = trivial_space(is), kws...)
  ITensorNetwork(itensor_constructor::Function, is::IndsNetwork; link_space = trivial_space(is), kwargs...)
  ```

* Construct an `ITensorNetwork` from an `IndsNetwork`. Initializes ITensors with `undef` storage on each vertex
  of the `IndsNetwork` with the corresponding indices (`itensornetwork.jl`):
  ```julia
  ITensorNetwork(eltype::Type, undef::UndefInitializer, is::IndsNetwork; kwargs...)
  ITensorNetwork(eltype::Type, is::IndsNetwork; kwargs...)
  ITensorNetwork(undef::UndefInitializer, is::IndsNetwork; kwargs...)
  ITensorNetwork(is::IndsNetwork; kwargs...)
  ```

## Global Operations on ITensorNetworks


## TreeTensorNetwork Constructors

* From `Op` and related types (`opsum_to_ttn.jl`):
  ```julia
  ttn(o::Op, s::IndsNetwork; kws...)
  ttn(o::Scaled{C, Op}, s::IndsNetwork; kws...)
  ttn(o::Sum{Op}, s::IndsNetwork; kws...)
  ttn(o::Prod{Op}, s::IndsNetwork; kws...)
  ttn(o::Scaled{C, Prod{Op}}, s::IndsNetwork; kws...)
  ttn(o::Sum{Scaled{C, Op}}, s::IndsNetwork; kws...)
  ```

## Miscellaneous Methods

* Methods in `partitioneditensornetwork.jl`.
  ```julia
  linkinds(pitn::PartitionedGraph, edge::QuotientEdge)
  ```
  To be revisited after Jack's work on NamedGraphs.

