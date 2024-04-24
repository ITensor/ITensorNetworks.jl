using ITensorNetworks
using Documenter

DocMeta.setdocmeta!(
  ITensorNetworks, :DocTestSetup, :(using ITensorNetworks); recursive=true
)

makedocs(;
  modules=[ITensorNetworks],
  authors="Matthew Fishman <mfishman@flatironinstitute.org>, Joseph Tindall <jtindall@flatironinstitute.org> and contributors",
  repo="https://github.com/mtfishman/ITensorNetworks.jl/blob/{commit}{path}#{line}",
  sitename="ITensorNetworks.jl",
  format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://mtfishman.github.io/ITensorNetworks.jl",
    assets=String[],
  ),
  pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/mtfishman/ITensorNetworks.jl", devbranch="main")
