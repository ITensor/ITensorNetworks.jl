module BaseExtensions
# Convert to real if possible
maybe_real(x::Real) = x
maybe_real(x::Complex) = iszero(imag(x)) ? real(x) : x

to_tuple(x) = (x,)
to_tuple(x::Tuple) = x
end
