#
# AbstractBijections
#

using Dictionaries

import Base: insert!, getindex, inv, length, show

abstract type AbstractBijection{D,I} end

_not_implemented() = error("Not implemented")

# Apply the Bijective function to an element
# `x ∈ domain(f)`, resulting in an element `y ∈ image(f)`.
apply(f::AbstractBijection, x) = _not_implemented()
inv(f::AbstractBijection) = _not_implemented()
domain(f::AbstractBijection) = _not_implemented()
image(f::AbstractBijection) = _not_implemented()
insert!(f::AbstractBijection, x, y) = _not_implemented()

apply_inv(f::AbstractBijection, y) = apply(inv(f), y)

getindex(f::AbstractBijection, x) = apply(f, x)
length(f::AbstractBijection) = length(domain(f))
domain_eltype(f::AbstractBijection) = eltype(domain(f))

function show(io::IO, mime::MIME"text/plain", f::AbstractBijection)
  println(io, typeof(f))
  for (x, y) in zip(domain(f), image(f))
    show(io, x)
    print(io, " ↔ ")
    show(io, y)
    println(io)
  end
  return nothing
end

show(io::IO, f::AbstractBijection) = show(io, MIME"text/plain"(), f)

struct Bijection{D,I} <: AbstractBijection{D,I}
  f::Dictionary{D,I}
  finv::Dictionary{I,D}
end

function Bijection(domain, image)
  f = Dictionary(domain, image)
  finv = Dictionary(image, domain)
  return Bijection(f, finv)
end
Bijection(image) = Bijection(eachindex(image), image)

function bijection(domain_image)
  f = dictionary(domain_image)
  finv = Dictionary(f.values, f.indices)
  return Bijection(f, finv)
end

apply(f::Bijection, x) = f.f[x]
inv(f::Bijection) = Bijection(f.finv, f.f)

domain(f::Bijection) = f.finv.values
image(f::Bijection) = f.f.values

function insert!(f::Bijection, x, y)
  insert!(f.f, x, y)
  insert!(f.finv, y, x)
  return f
end
