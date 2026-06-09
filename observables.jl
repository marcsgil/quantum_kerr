using LinearAlgebra, GeneralizedGrossPitaevskii, ProgressMeter, Reactant

function step_evolution(prob::GrossPitaevskiiProblem, tmax, observables, params; dt, nsaves)
    CUDA.GC.gc()
    CUDA.reclaim()
    alg = StrangSplitting()

    ΔT = tmax / nsaves
    iter = GeneralizedGrossPitaevskii.init(prob, alg, (0, ΔT); dt, nsaves=1, save_start=false, show_progress=false)

    prototype = observables(Reactant.to_rarray(prob.u0)..., params)
    observables_vals = [prototype for _ ∈ 0:nsaves]

    @showprogress for n ∈ 2:nsaves+1
        sol = dropdims.(GeneralizedGrossPitaevskii.solve!(iter)[2], dims=4)

        for (x_old, x_new) in zip(iter.u, sol)
            x_old .= x_new
        end

        observables_vals[n] = observables(Reactant.to_rarray(sol)..., params)
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

    H = dropdims(mean(Vα .* Vβ, dims=2), dims=2) - mean_Vα .* mean_Vβ

    v0α = view(Vα, 1, :)
    v1α = view(Vα, 2, :)
    v2α = view(Vα, 3, :)

    f00 = mean(v0α .^ 2, dims=1) - view(mean_Vα, 1:1) .^ 2
    f12 = mean(v1α .* v2α, dims=1) - view(mean_Vα, 2:2) .* view(mean_Vα, 3:3)

    vcat(f00, f12, H)
end

select_angle(ϕ) = ϕ > 0 ? (ϕ - π) / 2 : (ϕ + π) / 2

function compose_raw(raw)
    f00, f12, h00, h11, h22 = eachslice(stack(Array.(raw)), dims=1)
 
    λ₊ = @. 0.5 + real(h00) + abs(f00)
    λ₋ = @. 0.5 + real(h00) - abs(f00)
    duan = @. 2 * (1 + real(h11 + h22)) - 4abs(f12)

    ϕ_sq = @. select_angle(angle(f00))
    ϕ_duan = @. select_angle(angle(f12))

    ϕ_sq[1] = NaN
    ϕ_duan[1] = NaN

    λ₊, λ₋, duan, ϕ_sq, ϕ_duan
end

decibels(P, P0 = one(P) / 2) = 10 * log10(P / P0)