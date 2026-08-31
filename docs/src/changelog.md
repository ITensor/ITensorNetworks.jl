# Changelog

## [0.22.0](https://github.com/ITensor/ITensorNetworks.jl/compare/v0.21.5...release-0.22) - Unreleased

### Breaking changes

- Requires NamedGraphs v0.14. The network types here are `AbstractNamedGraph`
  subtypes, so its breaking changes carry through to them, including the output
  types of `vertices` and `edges` and the return values of the mutating
  functions. See the
  [NamedGraphs changelog](https://itensor.github.io/NamedGraphs.jl/stable/changelog/)
  ([#385](https://github.com/ITensor/ITensorNetworks.jl/pull/385)).
- The leaf type of a contraction sequence is `ITensorNetworks.Key`, where it was
  `NamedGraphs.Keys.Key`
  ([#385](https://github.com/ITensor/ITensorNetworks.jl/pull/385)).

### Non-breaking changes

- `gauge_walk` and the `edges` keyword of `combine_linkinds` take any collection
  of edges, where they previously required a `Vector` and so rejected the
  iterator that `edges` now returns
  ([#385](https://github.com/ITensor/ITensorNetworks.jl/pull/385)).
