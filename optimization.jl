# Based on P Hyllus and J Eisert 2006 New J. Phys. 8 51

using JuMP, Clarabel, LinearAlgebra, Combinatorics

function build_σ(n)
    σ₀ = [0 1;
        -1 0]
    σ = zeros(2n, 2n)
    for j ∈ 1:n
        σ[2j-1:2j, 2j-1:2j] .= σ₀
    end
    σ
end

"""
    validate_parties(parties::Vector{T}) where {T<:Vector}

Check if parties is a partition of 1:n for some n. If the check is succeeds, returns n.
"""
function validate_parties(parties::Vector{Vector{T}}) where {T<:Integer}
    x = sort(reduce(vcat, parties))
    for j ∈ eachindex(x)
        @assert x[j] == j
    end
    x[end]
end

function get_idx(modes)
    reduce(vcat, [[2j - 1, 2j] for j ∈ modes])
end

function block_diagonal(X, modes)
    idx = get_idx(modes)
    X[idx, idx]
end

function fullyWit(γ, parties=[[j] for j ∈ 1:size(γ, 1)÷2])
    n = validate_parties(parties)
    @assert size(γ, 1) == 2n
    @assert size(γ, 2) == 2n

    model = Model(Clarabel.Optimizer)
    σ = build_σ(n)

    @variable(model, X1[1:2n, 1:2n] in HermitianPSDCone())
    @variable(model, X2[1:2n, 1:2n] in HermitianPSDCone())

    @objective(model, Min, tr(real(X1) * γ))

    @constraint(model, tr(im * σ * X2) == -1)

    for party ∈ parties
        @constraint(model, real.(block_diagonal(X1, party)) .== real.(block_diagonal(X2, party)))
    end

    optimize!(model)

    if !is_solved_and_feasible(model)
        @warn("The model was not solved correctly.")
        return
    end

    objective_value(model), real(value(X1))
end

function multiWit(γ, parties=[[j] for j ∈ 1:size(γ, 1)÷2])
    n = validate_parties(parties)
    @assert size(γ, 1) == 2n
    @assert size(γ, 2) == 2n
    πs = partitions(1:length(parties), 2)
    K = length(πs)

    σ = build_σ(n)

    model = Model(Clarabel.Optimizer)

    @variable(model, X1[1:2n, 1:2n] in HermitianPSDCone())
    @objective(model, Min, tr(real(X1) * γ))

    Xkp1 = [@variable(model, [1:2n, 1:2n] in HermitianPSDCone()) for _ in 1:K]

    @variable(model, XKp2 ≥ 0)
    @variable(model, XKp3 ≥ 0)

    @variable(model, XKp3pk[1:K] >= 0)

    for (k, partition) ∈ enumerate(πs)
        @constraint(model, tr(im * σ * Xkp1[k]) + XKp2 - XKp3 + XKp3pk[k] == 0)
        for subset ∈ partition
            modes = reduce(vcat, parties[subset])
            @constraint(model, real.(block_diagonal(X1, modes)) .== real.(block_diagonal(Xkp1[k], modes)))
        end
    end

    @constraint(model, XKp2 - XKp3 == 1)

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

parties = [[1], [2]]

witness_val, Z = fullyWit(γ, parties)
##
witness_val, Z = multiWit(γ, parties)
##

γ = [2 0 0 0 1 0 0 0;
    0 1 0 0 0 0 0 -1;
    0 0 2 0 0 0 -1 0;
    0 0 0 1 0 -1 0 0;
    1 0 0 0 2 0 0 0;
    0 0 0 -1 0 4 0 0;
    0 0 -1 0 0 0 2 0
    0 -1 0 0 0 0 0 4]

parties = [[1, 2], [3, 4]]

witness_val, Z = fullyWit(γ, parties)
##
multiWit(γ, parties)
##
γ = [2 0 1 0 1 0;
    0 3 0 -1 0 -1;
    1 0 2 0 1 0;
    0 -1 0 3 0 -1;
    1 0 1 0 2 0
    0 -1 0 -1 0 3] / 2

witness_val, Z = fullyWit(γ)
##
witness_val, Z = multiWit(γ)

Z