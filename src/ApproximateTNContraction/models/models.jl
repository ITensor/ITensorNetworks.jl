module Models

export Model, critical_point, local_boltzmann_weight, mpo, localham, checklocalham

struct Model{model} end

Model(s::AbstractString) = Model{Symbol(s)}()

# For notation:
# Model"tfim" == Model{:tfim}
# Model"heisenberg" == Model{:heisenberg}
macro Model_str(s)
  return :(Model{$(Expr(:quote, Symbol(s)))})
end

include("ising_classical_2d.jl")
include("hamiltonians.jl")

end
