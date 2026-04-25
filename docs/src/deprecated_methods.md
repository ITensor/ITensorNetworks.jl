# Deprecated Methods

## ITensorNetwork Constructors

* From a named graph (forwards to construction from `IndsNetwork`).
  ```julia
  ITensorNetwork{V}(g::NamedGraph)
  ITensorNetwork(eltype::Type, undef::UndefInitializer, graph::AbstractNamedGraph; kws...)
  ITensorNetwork(f, graph::AbstractNamedGraph; kwargs...)
  ITensorNetwork(graph::AbstractNamedGraph; kwargs...)
  ```

* From a simple graph (forwards to construction from `IndsNetwork`).
  ```julia
  ITensorNetwork(eltype::Type, undef::UndefInitializer, graph::AbstractSimpleGraph; kws...)
  ITensorNetwork(f, graph::AbstractSimpleGraph; kwargs...)
  ITensorNetwork(graph::AbstractSimpleGraph; kwargs...)
  ```

* From a function over vertices or from a "value" (e.g. a string like `"Up"`,
  an `Op`, an array, or a per-vertex dict/array) that is converted to a callable and used
  to initialize each vertex tensor.
  ```julia
  ITensorNetwork(value, is::IndsNetwork; kwargs...)
  ITensorNetwork(elt::Type, f, is::IndsNetwork; link_space = trivial_space(is), kws...)
  ITensorNetwork(itensor_constructor::Function, is::IndsNetwork; link_space = trivial_space(is), kwargs...)
  ```

## Local Operations on ITensorNetworks

## Global Operations on ITensorNetworks

