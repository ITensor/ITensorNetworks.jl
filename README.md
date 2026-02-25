> [!WARNING]
> This is a pre-release software. There are no guarantees that functionality won't break
> from version to version, though we will try our best to indicate breaking changes
> following [semantic versioning](https://semver.org/) (semver) by bumping the minor
> version of the package. We are biasing heavily towards "moving fast and breaking things"
> during this stage of development, which will allow us to more quickly develop the package
> and bring it to a point where we have enough features and are happy enough with the external
> interface to officially release it for general public use.
>
> In short, use this package with caution, and don't expect the interface to be stable
> or for us to clearly announce parts of the code we are changing.

# ITensorNetworks.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://itensor.github.io/ITensorNetworks.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://itensor.github.io/ITensorNetworks.jl/dev/)
[![Build Status](https://github.com/ITensor/ITensorNetworks.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/ITensorNetworks.jl/actions/workflows/Tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/ITensorNetworks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensorNetworks.jl)
[![Code Style](https://img.shields.io/badge/code_style-ITensor-purple)](https://github.com/ITensor/ITensorFormatter.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Support

<picture>
  <source media="(prefers-color-scheme: dark)" width="20%" srcset="docs/src/assets/CCQ-dark.png">
  <img alt="Flatiron Center for Computational Quantum Physics logo." width="20%" src="docs/src/assets/CCQ.png">
</picture>


ITensorNetworks.jl is supported by the Flatiron Institute, a division of the Simons Foundation.

## Installation instructions

This package can be added as usual through the package manager:

```julia
julia> using Pkg: Pkg

julia> Pkg.add("ITensorNetworks")
```

## Examples

````julia
using ITensorNetworks: ITensorNetworks
````

Examples go here.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

