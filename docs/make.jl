using Documenter: Documenter, DocMeta, deploydocs, makedocs
using Graphs: Graphs
using ITensorFormatter: ITensorFormatter
using ITensorNetworks: ITensorNetworks
using ITensors: ITensors
using LinearAlgebra: LinearAlgebra
using OMEinsumContractionOrders
using TensorOperations

DocMeta.setdocmeta!(
    ITensorNetworks,
    :DocTestSetup,
    quote
        using Graphs: dst, edges, src, vertices
        using ITensorNetworks
        using ITensorNetworks:
            TreeTensorNetwork, expect, loginner, mps, orthogonalize, siteinds, truncate, ttn
        using ITensors: inner
        using LinearAlgebra: norm, normalize
        using OMEinsumContractionOrders
        using TensorOperations
    end;
    recursive = true
)

ITensorFormatter.make_index!(pkgdir(ITensorNetworks))

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
            "ITensor Networks" => "itensor_networks.md",
            "Tree Tensor Networks" => "tree_tensor_networks.md",
            "Computing Properties" => "computing_properties.md",
            "Solvers" => "solvers.md",
        ],
        "API Reference" => "reference.md",
    ]
)

deploydocs(;
    repo = "github.com/ITensor/ITensorNetworks.jl", devbranch = "main", push_preview = true
)
