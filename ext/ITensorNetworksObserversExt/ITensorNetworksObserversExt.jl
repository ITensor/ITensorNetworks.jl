module ITensorNetworksObserversExt
using ITensorNetworks: ITensorNetworks
using Observers.DataFrames: AbstractDataFrame
using Observers: Observers

function ITensorNetworks.update_observer!(observer::AbstractDataFrame; kwargs...)
    return Observers.update!(observer; kwargs...)
end
end
