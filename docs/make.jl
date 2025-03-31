using ITensorNetworks: ITensorNetworks
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(
  ITensorNetworks, :DocTestSetup, :(using ITensorNetworks); recursive=true
)

include("make_index.jl")

makedocs(;
  modules=[ITensorNetworks],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="ITensorNetworks.jl",
  format=Documenter.HTML(;
    canonical="https://itensor.github.io/ITensorNetworks.jl",
    edit_link="main",
    assets=["assets/favicon.ico", "assets/extras.css"],
  ),
  pages=["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo="github.com/ITensor/ITensorNetworks.jl", devbranch="main", push_preview=true
)
