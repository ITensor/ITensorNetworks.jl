using Observers: Observers

"""
Overload of `Observers.update!`.
"""
Observers.update!(::Nothing; kwargs...) = nothing
