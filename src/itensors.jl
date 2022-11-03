# Tensor sum: `A ⊞ B = A ⊗ Iᴮ + Iᴬ ⊗ B`
# https://github.com/JuliaLang/julia/issues/13333#issuecomment-143825995
# "PRESERVATION OF TENSOR SUM AND TENSOR PRODUCT"
# C. S. KUBRUSLY and N. LEVAN
# https://www.emis.de/journals/AMUC/_vol-80/_no_1/_kubrusly/kubrusly.pdf
function tensor_sum(A::ITensor, B::ITensor)
  extend_A = filterinds(uniqueinds(B, A); plev=0)
  extend_B = filterinds(uniqueinds(A, B); plev=0)
  for i in extend_A
    A *= op("I", i)
  end
  for i in extend_B
    B *= op("I", i)
  end
  return A + B
end
