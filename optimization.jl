# Based on P Hyllus and J Eisert 2006 New J. Phys. 8 51

using JuMP, Clarabel, LinearAlgebra

function build_σ(n)
    σ₀ = [0 1;
        -1 0]
    σ = zeros(2n, 2n)
    for j ∈ 1:n
        σ[2j-1:2j, 2j-1:2j] .= σ₀
    end
    σ
end

function fullyWit(γ, ns)
    n = sum(ns)
    @assert size(γ, 1) == 2n
    @assert size(γ, 2) == 2n

    model = Model(Clarabel.Optimizer)
    σ = build_σ(n)

    @variable(model, X1[1:2n, 1:2n] in HermitianPSDCone())
    @variable(model, X2[1:2n, 1:2n] in HermitianPSDCone())

    @objective(model, Min, tr(real(X1) * γ))

    @constraint(model, tr(im * σ * X2) == -1)

    start = 1
    for m ∈ ns
        stop = start + 2m - 1
        @constraint(model, real.(X1[start:stop, start:stop]) .== real.(X2[start:stop, start:stop]))
        start = stop + 1
    end

    optimize!(model)

    if !is_solved_and_feasible(model)
        @warn("The model was not solved correctly.")
        return
    end

    objective_value(model), real(value(X1))
end
##
r = 0.1

γ = [cosh(r) 0 sinh(r) 0;
    0 cosh(r) 0 -sinh(r);
    sinh(r) 0 cosh(r) 0;
    0 -sinh(r) 0 cosh(r)]

ns = [1, 1]

witness_val, Z = fullyWit(γ, ns)

Z
##

γ = [2 0 0 0 1 0 0 0;
    0 1 0 0 0 0 0 -1;
    0 0 2 0 0 0 -1 0;
    0 0 0 1 0 -1 0 0;
    1 0 0 0 2 0 0 0;
    0 0 0 -1 0 4 0 0;
    0 0 -1 0 0 0 2 0
    0 -1 0 0 0 0 0 4]

ns = [2, 2]

witness_val, Z = fullyWit(γ, ns)