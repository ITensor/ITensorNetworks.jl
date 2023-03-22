
struct TDVPOrder{order} end

TDVPOrder(order::Int) = TDVPOrder{order}()

directions(::TDVPOrder) = error("Not implemented")
sub_time_steps(::TDVPOrder) = error("Not implemented")

# TODO: make these both length 1 for order 1?
directions(::TDVPOrder{1}) = [Base.Forward, Base.Reverse]
sub_time_steps(::TDVPOrder{1}) = [1.0, 0.0]

directions(::TDVPOrder{2}) = [Base.Forward, Base.Reverse]
sub_time_steps(::TDVPOrder{2}) = [1/2, 1/2]

directions(::TDVPOrder{4}) = [Base.Forward, Base.Reverse,Base.Forward,Base.Reverse]
function sub_time_steps(::TDVPOrder{4})
  s = 1.0/(2 - 2^(1/3))
  return [s/2, s/2,(1-2*s)/2,(1-2*s)/2, s/2, s/2]
end

function one_site_update_sweep(tdvp_order::TDVPOrder, time_step::Number, graph::AbstractGraph; kwargs...)
  time_sign(loc) = (loc isa AbstractEdge) ? -1 : +1
  sweep = nothing
  for (n,dir) in enumerate(directions(tdvp_order))
    fac = sub_time_steps(tdvp_order)[n]
    substep = [
             (loc, (; substep=n, time_step=time_step*fac*time_sign(loc))) for loc in one_site_half_sweep(dir, graph; reverse_step=true, kwargs...)
    ]
    sweep = isnothing(sweep) ? substep : vcat(sweep,substep)
  end
  return sweep
end

function two_site_update_sweep(tdvp_order::TDVPOrder, time_step::Number, graph::AbstractGraph; kwargs...)
  time_sign(loc) = (length(loc)==1) ? -1 : +1
  sweep = nothing
  for (n,dir,fac) in enumerate(zip(directions(tdvp_order),sub_time_steps(tdvp_order)))
    substep = [
               (loc, (; substep=n, time_step=time_step*fac*time_sign(loc))) for loc in two_site_half_sweep(dir, graph; reverse_step=true, kwargs...)
    ]
    sweep = isnothing(sweep) ? substep : vcat(sweep,substep)
  end
  return sweep
end
