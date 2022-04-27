#' # ITensorNetworks
#'
#' A package to provide general network data structures and tools to use with ITensors.jl.

#' ## Installation
#'
#' This package relies on a few unregistered packages. To install, you will need to do:
#'
#' ```julia
#' julia> using Pkg
#' 
#' julia> Pkg.add(url="https://github.com/mtfishman/MultiDimDictionaries.jl")
#' 
#' julia> Pkg.add(url="https://github.com/mtfishman/NamedGraphs.jl")
#' 
#' julia> Pkg.add(url="https://github.com/mtfishman/DataGraphs.jl")
#' 
#' julia> Pkg.add(url="https://github.com/mtfishman/ITensorNetworks.jl")
#' ```

#' ## Examples
#'
#' Here are is an example of making a tensor network on a chain graph (a tensor train or matrix product state):
#+ term=true

using ITensors
using ITensorNetworks
tn = ITensorNetwork(named_grid((4,)); link_space=2)
tn[1]
tn[2]
neighbors(tn, 1)
neighbors(tn, 2)
neighbors(tn, 3)
neighbors(tn, 4)

#' and here is a similar example for making a tensor network on a grid (a tensor product state or project entangled pair state (PEPS)):
#+ term=true

tn = ITensorNetwork(named_grid((2, 2)); link_space=2)
tn[1, 1]
neighbors(tn, 1, 1)
neighbors(tn, 1, 2)
tn_1 = tn[1, :]
tn_2 = tn[2, :]

#' Networks can also be merged/unioned:
#+ term=true

using ITensorUnicodePlots
s = siteinds("S=1/2", named_grid((3,)))
tn1 = ITensorNetwork(s; link_space=2)
tn2 = ITensorNetwork(s; link_space=2)
@visualize tn1;
@visualize tn2;
Z = prime(tn1; sites=[]) ⊗ tn2;
@visualize Z;
using ITensors.ContractionSequenceOptimization
optimal_contraction_sequence(Z)
Z̃ = contract(Z, (1, 1) => (2, 1));
@visualize Z̃;

#' ## Generating this README
#'
#' This file was generated with [weave.jl](https://github.com/JunoLab/Weave.jl) with the following commands:
#' ```julia
#' using ITensorNetworks, Weave
#' weave(joinpath(pkgdir(ITensorNetworks), "examples", "README.jl"); doctype="github", out_path=pkgdir(ITensorNetworks))
#' ```
