struct TDVPOrder{order,direction} end

TDVPOrder(order::Int, direction::Base.Ordering) = TDVPOrder{order,direction}()

orderings(::TDVPOrder) = error("Not implemented")
sub_time_steps(::TDVPOrder) = error("Not implemented")

function orderings(::TDVPOrder{1,direction}) where {direction}
  return [direction, Base.ReverseOrdering(direction)]
end
sub_time_steps(::TDVPOrder{1}) = [1.0, 0.0]

function orderings(::TDVPOrder{2,direction}) where {direction}
  return [direction, Base.ReverseOrdering(direction)]
end
sub_time_steps(::TDVPOrder{2}) = [1.0 / 2.0, 1.0 / 2.0]

function orderings(::TDVPOrder{4,direction}) where {direction}
  return [direction, Base.ReverseOrdering(direction)]
end
function sub_time_steps(::TDVPOrder{4})
  s = 1.0 / (2.0 - 2.0^(1.0 / 3.0))
  return [s / 2.0, s / 2.0, (1.0 - 2.0 * s) / 2.0, (1.0 - 2.0 * s) / 2.0, s / 2.0, s / 2.0]
end
