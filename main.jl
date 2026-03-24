using GeneralizedGrossPitaevskii, StructuredLight, CairoMakie, CUDA, Statistics
includet("observables.jl")

function dispersion(k, param)
    factor = (k[1]^2 + k[2]^2) / 4
    @SVector [factor, -factor]
end

function nonlinearity(u, param)
    factor = param.G * u[1] * u[2]
    @SVector [factor, -factor]
end

function position_noise_func(u, r, param)
    factor = √(im * param.G)
    @SVector [factor * u[1], -conj(factor) * u[2]]
end

L = 6f0
lengths = (L, L)
N = 256
dr = L / N
dA = dr^2
rs = LinRange(-L / 2, L / 2 - dr, N)
α = 120f0

u0 = lg(rs, rs, l=0) |> cu

g_eff = 1f-3
G = g_eff / (4 * dA)

U0 = (α * u0, conj(α * u0))
noise_prototype = similar.(U0, Float32)
param = (; G)

prob = GrossPitaevskiiProblem(U0, lengths; dispersion, nonlinearity, position_noise_func, noise_prototype, param)
alg = StrangSplitting()
tspan = (0, 4f-2)
nsaves = 128
dt = tspan[end] / 256

ts, sol = solve(prob, alg, tspan; dt, nsaves, save_start=false)

save_animation(Array(abs2.(sol[1])), "test.mp4")
##
u0_many = stack(u0 for _ ∈ 1:512) |> cu
U0 = (α * u0_many, conj(α * u0_many))
noise_prototype = similar.(U0, Float32)
prob = GrossPitaevskiiProblem(U0, lengths; dispersion, nonlinearity, position_noise_func, noise_prototype, param)

v = lg(rs, rs, l=0) |> cu
v1 = cis(π / 4) * lg(rs, rs, l=1) |> cu
v2 = cis(π / 4) * lg(rs, rs, l=-1) |> cu

observables = (
    (α, β) -> correlation(α, β, v, v, dA),
    (α, β) -> correlation(α, β, -im * v, -im * v, dA),
    (α, β) -> correlation(α, β, v, -im * v, dA),
    (α, β) -> correlation(α, β, v1, v1, dA),
    (α, β) -> correlation(α, β, v1, v2, dA),
    (α, β) -> correlation(α, β, v2, v2, dA),
    (α, β) -> correlation(α, β, -im * v1, -im * v1, dA),
    (α, β) -> correlation(α, β, -im * v1, -im * v2, dA),
    (α, β) -> correlation(α, β, -im * v2, -im * v2, dA),
)

ts, observables_vals = step_evolution(prob, tspan[end], observables; dt, nsaves=64)
##
duan = duan_criterion(ntuple(i -> observables_vals[i+3, :], 6)..., -1)

R12 = imag.(sum(conj.(v1 .* v2) .* u0 .^ 2) * dA)

λ₊, λ₋, θ = diagonalize_correlation(ntuple(i -> observables_vals[i, :], 3)...)

observables_vals[2, :]
lines(ts, θ)


with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=18, size=(900, 600))

    ax1 = Axis(fig[1, 1], xlabel=L"z/z_R")
    lines!(ax1, ts, real.(observables_vals[1, :]), label=L"\langle \Delta X^2 \rangle", linewidth=4)
    lines!(ax1, ts, real.(observables_vals[2, :]), label=L"\langle \Delta P^2 \rangle", linewidth=4)
    axislegend(ax1, position=:lt)

    ax2 = Axis(fig[1, 2], xlabel=L"z/z_R")
    lines!(ax2, ts, λ₋, label=L"\lambda_-", linewidth=4)
    lines!(ax2, ts, λ₊, label=L"\lambda_+", linewidth=4)
    axislegend(ax2, position=:lt)

    ax3 = Axis(fig[2, 1], ylabel=L"\theta", xlabel=L"z/z_R")
    lines!(ax3, ts, θ, label=L"\theta", linewidth=4)

    ax3 = Axis(fig[2, 2], ylabel=L"\langle \Delta (X_1 \pm X_2) + \Delta (P_1 \mp P_2)\rangle", xlabel=L"z/z_R")
    lines!(ax3, ts, duan, label="Positive P", linewidth=4)
    hlines!(ax3, [2], label="Duan bound", linestyle=:dash, color=:red, linewidth=4)
    lines!(ax3, ts, 2 .- ts .* R12 * g_eff * α^2, label="Linear Theory", linestyle=:dot, linewidth=4, color=:green)
    axislegend(ax3, position=:lt)
    save("Plots/duan.png", fig)
    fig
end