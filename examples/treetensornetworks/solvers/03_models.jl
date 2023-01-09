using ITensors

function heisenberg(n; J=1.0, J2=0.0)
  ℋ = OpSum()
  if !iszero(J)
    for j in 1:(n - 1)
      ℋ += J / 2, "S+", j, "S-", j + 1
      ℋ += J / 2, "S-", j, "S+", j + 1
      ℋ += J, "Sz", j, "Sz", j + 1
    end
  end
  if !iszero(J2)
    for j in 1:(n - 2)
      ℋ += J2 / 2, "S+", j, "S-", j + 2
      ℋ += J2 / 2, "S-", j, "S+", j + 2
      ℋ += J2, "Sz", j, "Sz", j + 2
    end
  end
  return ℋ
end
