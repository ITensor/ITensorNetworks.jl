# Deprecated Methods

## ITensorNetwork Constructors

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

## Local Operations on ITensorNetworks

* Combine (fuse) every link index of a tensor network, or a chosen set of edges, into
  a single index per edge using `combiner` tensors.
  Not used anywhere in the library? (`abstractitensornetwork.jl`):
  ```julia
  linkinds_combiners(tn::AbstractITensorNetwork; edges = edges(tn))
  combine_linkinds(tn::AbstractITensorNetwork, combiners)
  combine_linkinds(tn::AbstractITensorNetwork; edges = edges(tn))
  ```

* Functions in `apply.jl` which are unused, even inside that file (`apply.jl`):
  ```julia
  _gate_vertices(o::ITensor, ψ)
  _gate_vertices(o::AbstractEdge, ψ)
  _contract_gate(o::ITensor, ψv1, Λ, ψv2)
  _contract_gate(o::AbstractEdge, ψv1, Λ, ψv2)
  ```

## Global Operations on ITensorNetworks

* Scale tensors at chosen vertices by per-vertex weights, either out-of-place or in-place (`abstractitensornetwork.jl`):
  ```julia
  scale(tn::AbstractITensorNetwork, vertices_weights::Dictionary; kwargs...)
  scale(weight_function::Function, tn; kwargs...)
  scale!(tn::AbstractITensorNetwork, vertices_weights::Dictionary)
  scale!(weight_function::Function, tn::AbstractITensorNetwork; kwargs...)
  ```

