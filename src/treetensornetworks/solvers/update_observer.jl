function Observers.update!(observer::ITensors.AbstractObserver; kwargs...)
  return measure!(observer; kwargs...)
end
