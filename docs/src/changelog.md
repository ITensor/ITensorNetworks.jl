# Changelog

## [0.23.0](https://github.com/ITensor/ITensorNetworks.jl/compare/v0.22.0...main) - Unreleased

### Breaking changes

- Requires NamedGraphs v0.14. The network types here are `AbstractNamedGraph`
  subtypes, so its breaking changes carry through to them, including the output
  types of `vertices` and `edges` and the return values of the mutating
  functions. See the
  [NamedGraphs changelog](https://itensor.github.io/NamedGraphs.jl/stable/changelog/)
  ([#384](https://github.com/ITensor/ITensorNetworks.jl/pull/384)).
- The leaf type of a contraction sequence is `ITensorNetworks.Key`, where it was
  `NamedGraphs.Keys.Key`
  ([#384](https://github.com/ITensor/ITensorNetworks.jl/pull/384)).
