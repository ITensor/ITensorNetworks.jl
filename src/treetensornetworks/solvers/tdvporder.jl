
struct TDVPOrder{order} end

TDVPOrder(order::Int) = TDVPOrder{order}()

directions(::TDVPOrder) = error("Not implemented")
sub_time_steps(::TDVPOrder) = error("Not implemented")

directions(::TDVPOrder{1}) = [Base.Forward, Base.Reverse]
sub_time_steps(::TDVPOrder{1}) = [1.0, 0.0]

directions(::TDVPOrder{2}) = [Base.Forward, Base.Reverse]
sub_time_steps(::TDVPOrder{2}) = [1/2, 1/2]

directions(::TDVPOrder{4}) = [Base.Forward, Base.Reverse, Base.Forward, Base.Reverse]
function sub_time_steps(::TDVPOrder{4})
  s = 1.0/(2 - 2^(1/3))
  return [s/2, s/2,(1-2*s)/2,(1-2*s)/2, s/2, s/2]
end

function tdvp_one_site_region(edge; last_edge=false, substep, time_step, kwargs...)
  site = ([src(edge)], (; substep, time_step))
  bond = (edge, (; substep, time_step=-time_step))
  if last_edge
    return site, bond, ([dst(edge)], (; substep, time_step))
  end
  return site, bond
end

function tdvp_two_site_region(edge; last_edge=false, substep, time_step, kwargs...)
  site2 = ([src(edge),dst(edge)], (; substep, time_step))
  site1 = ([dst(edge)], (; substep, time_step = -time_step))
  if last_edge
    return (site2,)
  end
  return site2, site1
end

function tdvp_sweep(order::Int, nsite::Int, time_step::Number, graph::AbstractGraph; kwargs...)
  tdvp_order = TDVPOrder(order)

  half_sweep_func = nothing
  if nsite == 1
    region_function = tdvp_one_site_region
  elseif nsite == 2
    region_function = tdvp_two_site_region
  else
    error("nsite=$nsite not supported in tdvp")
  end

  sweep = []
  for (substep,(dir,fac)) in enumerate(zip(directions(tdvp_order),sub_time_steps(tdvp_order)))
    half = half_sweep(dir,graph,region_function; substep, time_step=time_step*fac, kwargs...)
    append!(sweep,half)
  end

  return sweep
end
