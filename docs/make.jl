using Documenter: Documenter, DocMeta, deploydocs, makedocs
using Graphs: Graphs
using ITensorNetworks: ITensorNetworks
using ITensors: ITensors
using LinearAlgebra: LinearAlgebra

DocMeta.setdocmeta!(
    ITensorNetworks, :DocTestSetup, :(using ITensorNetworks); recursive = true
)

include("make_index.jl")

makedocs(;
    modules = [ITensorNetworks],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "ITensorNetworks.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/ITensorNetworks.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"]
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Tensor Networks" => "tensor_networks.md",
            "Tree Tensor Networks" => "tree_tensor_networks.md",
            "Computing Properties" => "computing_properties.md",
            "Solvers" => "solvers.md",
        ],
        "API Reference" => "reference.md",
    ],
    warnonly = true
)

deploydocs(;
    repo = "github.com/ITensor/ITensorNetworks.jl", devbranch = "main", push_preview = true
)
