function update_observer!(observer; kwargs...)
  return error("Not implemented")
end

# Default fallback if no observer is specified.
# Makes the source code a bit simpler, though
# maybe it is a bit too "tricky" and should be
# removed.
function update_observer!(observer::Nothing; kwargs...)
  return nothing
end

struct ComposedObservers{Observers<:Tuple}
  observers::Observers
end
compose_observers(observers...) = ComposedObservers(observers)
function update_observer!(observer::ComposedObservers; kwargs...)
  for observerᵢ in observer.observers
    update_observer!(observerᵢ; kwargs...)
  end
  return observer
end

struct ValuesObserver{Values<:NamedTuple}
  values::Values
end
function update_observer!(observer::ValuesObserver; kwargs...)
  for key in keys(observer.values)
    observer.values[key][] = kwargs[key]
  end
  return observer
end
