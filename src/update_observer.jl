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
