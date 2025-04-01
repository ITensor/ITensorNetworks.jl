using Documenter: Documenter, DocMeta, deploydocs, makedocs
using ITensorNetworks: ITensorNetworks
using Literate: Literate

DocMeta.setdocmeta!(
  ITensorNetworks, :DocTestSetup, :(using ITensorNetworks); recursive=true
)

include("make_index.jl")

Literate.markdown(
  joinpath(@__DIR__, "src", "examples.jl"),
  joinpath(@__DIR__, "src");
  flavor=Literate.DocumenterFlavor(),
)

makedocs(;
  modules=[ITensorNetworks],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="ITensorNetworks.jl",
  format=Documenter.HTML(;
    canonical="https://itensor.github.io/ITensorNetworks.jl",
    edit_link="main",
    assets=["assets/favicon.ico", "assets/extras.css"],
  ),
  pages=["Home" => "index.md", "Examples" => "examples.md", "Reference" => "reference.md"],
  warnonly=true,
)

deploydocs(;
  repo="github.com/ITensor/ITensorNetworks.jl", devbranch="main", push_preview=true
)
