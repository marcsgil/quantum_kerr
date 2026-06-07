using LinearAlgebra, GeneralizedGrossPitaevskii, ProgressMeter, Reactant

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
    projection(v1, v2, dA) / 2 + real(
        expval_annihilation_annihilation(α, β, v1, v2, dA) - expval_annihilation(α, β, v1, dA) * expval_annihilation(α, β, v2, dA)
        +
        expval_creation_annihilation(α, β, v1, v2, dA) - expval_annihilation(α, β, v1, dA) * expval_creation(α, β, v2, dA)
    )
end

function step_evolution(prob::GrossPitaevskiiProblem, tmax, observables, params; dt, nsaves)
    CUDA.GC.gc()
    CUDA.reclaim()
    alg = StrangSplitting()

    ΔT = tmax / nsaves
    iter = GeneralizedGrossPitaevskii.init(prob, alg, (0, ΔT); dt, nsaves=1, save_start=false, show_progress=false)

    prototype = observables(Reactant.to_rarray(prob.u0)..., params)
    observables_vals = similar(prototype, length(prototype), nsaves + 1)
    observables_vals[:, 1] .= prototype

    @showprogress for slice ∈ eachslice((@view observables_vals[:, 2:end]), dims=2)
        sol = dropdims.(GeneralizedGrossPitaevskii.solve!(iter)[2], dims=4)

        for (x_old, x_new) in zip(iter.u, sol)
            x_old .= x_new
        end

        slice .= observables(Reactant.to_rarray(sol)..., params)
    end

    (0:nsaves) .* ΔT, observables_vals
end

function raw_observables(α, β, V)
    N = size(α, 3)

    α = reshape(α, :, N)
    β = reshape(β, :, N)

    Vα = V' * α
    Vβ = transpose(V) * β

    mean_Vα = dropdims(mean(Vα, dims=2), dims=2)
    mean_Vβ = dropdims(mean(Vβ, dims=2), dims=2)


    ΔaV² = mean(Vα .^ 2, dims=2) - mean_Vα .^ 2
    ΔaVᵈaV = mean(Vα .* Vβ, dims=2) - mean_Vα .* mean_Vβ
    vec(vcat(ΔaV², ΔaVᵈaV))
end

function compose_raw(raw)
    raw = Array(raw)
    ΔX² = @. 0.5 + real(raw[1, :] + raw[4, :])
    ΔP² = @. 0.5 + real(-raw[1, :] + raw[4, :])
    duan = @. 2 + real(raw[2, :] + raw[5, :] - raw[3, :] + raw[6, :])
    ΔX², ΔP², duan
end