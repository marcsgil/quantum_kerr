using LinearAlgebra, GeneralizedGrossPitaevskii, ProgressMeter

function projection(u, v, dA)
    (u ⋅ v) * dA
end

function expval_annihilation(α, β, v, dA)
    mapreduce(α -> (v ⋅ α) * dA, +, eachslice(α, dims=3)) / size(α, 3)
end

function expval_creation(α, β, v, dA)
    conjv = conj.(v)
    mapreduce(β -> (conjv ⋅ β) * dA, +, eachslice(β, dims=3)) / size(β, 3)
end

function expval_creation_annihilation(α, β, v1, v2, dA)
    conjv2 = conj.(v2)
    mapreduce((α, β) -> (v1 ⋅ α) * (conjv2 ⋅ β) * dA^2, +, eachslice(α, dims=3), eachslice(β, dims=3)) / size(α, 3)
end

function expval_annihilation_annihilation(α, β, v1, v2, dA)
    mapreduce(α -> (v1 ⋅ α) * (v2 ⋅ α) * dA^2, +, eachslice(α, dims=3)) / size(α, 3)
end

function correlation(α, β, v1, v2, dA)
    real(projection(v1, v2, dA) / 2 +
         expval_annihilation_annihilation(α, β, v1, v2, dA) - expval_annihilation(α, β, v1, dA) * expval_annihilation(α, β, v2, dA)
         +
         expval_creation_annihilation(α, β, v1, v2, dA) - expval_annihilation(α, β, v1, dA) * expval_creation(α, β, v2, dA)
    )
end

function step_evolution(prob::GrossPitaevskiiProblem, tmax, observables; dt, nsaves)
    alg = StrangSplitting()

    ΔT = tmax / nsaves
    ts = (1:nsaves) .* ΔT
    observables_vals = Matrix{eltype(dt)}(undef, length(observables), nsaves + 1)

    for (j, obs) in enumerate(observables)
        observables_vals[j, 1] = obs(prob.u0...)
    end

    @showprogress for (i, T) in enumerate(ts)
        tspan = (T - ΔT, T)
        sol = solve(prob, alg, tspan; dt, nsaves=1, save_start=false, show_progress=false)[2]

        for (x_old, x_new) in zip(prob.u0, sol)
            x_old .= x_new
        end

        # for (j, obs) in enumerate(observables)
        #     observables_vals[j, i+1] = obs(sol...)
        # end

        observables_vals[:, i+1] .= map(obs -> obs(sol...), observables)
    end

    (0:nsaves) .* ΔT, observables_vals
end

function duan_criterion(X1_2, X1P1, P1_2, X2_2, X2P2, P2_2, sign)
    @. X1_2 + 2 * sign * X1P1 + P1_2 + X2_2 - 2 * sign * X2P2 + P2_2
end

function diagonalize_correlation(X2, P2, XP)
    λ₊ = @. (X2 + P2) / 2 + sqrt(((X2 - P2) / 2)^2 + XP^2)
    λ₋ = @. (X2 + P2) / 2 - sqrt(((X2 - P2) / 2)^2 + XP^2)
    θ = @. atan(λ₋ - X2, XP)

    λ₊, λ₋, θ
end