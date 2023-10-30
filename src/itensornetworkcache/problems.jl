abstract type AbstractProblem end

# A * x ∼ x * λ
struct EigenProblem{Operator} <: AbstractProblem
  operator::Operator
end

function cache(problem::EigenProblem, state::AbstractITensorNetwork; contract_alg=default_contract_alg())
  return EigenProblem(rayleigh_quotient_cache(problem.operator, state; contract_alg))
end

# A * x ∼ b
struct LinearProblem{Operator,ConstantTerm} <: AbstractProblem
  operator::Operator
  constant_term::ConstantTerm
end

# A * x ∼ ∂ₜx
struct LinearPDEProblem{Operator} <: AbstractProblem
  operator::Operator
end
