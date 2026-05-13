module ITensorNetworksITensorVisualizationBaseExt

using Graphs: vertices
using ITensorNetworks: AbstractITensorNetwork
using ITensorVisualizationBase: ITensorVisualizationBase

function ITensorVisualizationBase.visualize(
        tn::AbstractITensorNetwork,
        args...;
        vertex_labels_prefix = nothing,
        vertex_labels = nothing,
        kwargs...
    )
    if !isnothing(vertex_labels_prefix)
        vertex_labels = [vertex_labels_prefix * string(v) for v in vertices(tn)]
    end
    return ITensorVisualizationBase.visualize(
        [tn[v] for v in vertices(tn)], args...; vertex_labels, kwargs...
    )
end

end
