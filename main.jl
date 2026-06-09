using GeneralizedGrossPitaevskii, StructuredLight, CairoMakie, CUDA, Statistics, Reactant
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

L = 8e0
lengths = (L, L)
N = 128
dr = L / N
dA = dr^2
rs = LinRange(-L / 2, L / 2 - dr, N)
α = 10e0

u0 = lg(rs, rs, l=0)

g = -4e-3
G = g / (4 * dA)

U0 = (α * u0, conj(α * u0))
noise_prototype = similar.(U0, Float64)
param = (; G)

prob = GrossPitaevskiiProblem(U0, lengths; dispersion, nonlinearity, position_noise_func, noise_prototype, param)
alg = StrangSplitting()
tspan = (0, 4e-2)
nsaves = 128
dt = tspan[end] / 128
##
zs, sol = solve(prob, alg, tspan; dt, nsaves, save_start=false)

save_animation(Array(abs2.(sol[1])), "test.mp4")
##
u0_many = stack(u0 for _ ∈ 1:1024) |> CuArray
U0 = (α * u0_many, conj(α * u0_many))
noise_prototype = similar.(U0, Float64)
prob = GrossPitaevskiiProblem(U0, lengths; dispersion, nonlinearity, position_noise_func, noise_prototype, param)

v = lg(rs, rs, l=0)
v1 = lg(rs, rs, l=1)
v2 = lg(rs, rs, l=-1)

V = hcat(vec.((v, v1, v2))...) * dA

CUDA.reclaim()
CUDA.GC.gc()

rV = Reactant.to_rarray(V)
rU0 = Reactant.to_rarray.(U0)

f = @compile raw_observables(rU0..., rV)

zs, observables_vals = step_evolution(prob, tspan[end], f, rV; dt, nsaves=32)

λ₊, λ₋, duan, ϕ_sq, ϕ_duan = compose_raw(observables_vals)


R12 = (v1 .* v2) ⋅ (u0 .^ 2) * dA
R00 = (v .^ 2) ⋅ (u0 .^ 2) * dA

D_opt_linear = @. 2 - zs * abs(g * α^2 * R12)
λ₋_linear = @. 0.5 - zs * abs(g * α^2 * R00) / 4


with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=18, size=(1200, 600))

    ax1 = Axis(fig[1, 1], ylabel="Quadrature Variance (dB)")
    lines!(ax1, zs, decibels.(λ₋), label=L"\lambda_-", linewidth=4)
    lines!(ax1, zs, decibels.(λ₋_linear), label="Linear Theory", linestyle=:dot, linewidth=4, color=:black)
    axislegend(ax1, position=:lb)
    hidexdecorations!(ax1, ticks=false, grid=false)

    ax2 = Axis(fig[1, 2], ylabel=L"D_{\text{opt}} / D_0")
    lines!(ax2, zs, real.(duan) / 2, label="Positive P", linewidth=4)
    lines!(ax2, zs, D_opt_linear / 2, label="Linear Theory", linestyle=:dot, linewidth=4, color=:black)
    axislegend(ax2, position=:lb)
    hidexdecorations!(ax2, ticks=false, grid=false)

    ax3 = Axis(fig[2, 1], ylabel="Squeezing angle", xlabel=L"z/z_R", yticks=([-π / 2, -π / 4, 0, π / 4, π / 2], [L"-π/2", L"-π/4", L"0", L"π/4", L"π/2"]))
    scatter!(ax3, zs, ϕ_sq)
    ylims!(ax3, -π / 2, π / 2)

    ax4 = Axis(fig[2, 2], ylabel="Optimal Duan angle", xlabel=L"z/z_R", yticks=([-π / 2, -π / 4, 0, π / 4, π / 2], [L"-π/2", L"-π/4", L"0", L"π/4", L"π/2"]))
    scatter!(ax4, zs, ϕ_duan)
    ylims!(ax4, -π / 2, π / 2)

    linkxaxes!(ax1, ax3)
    linkxaxes!(ax2, ax4)

    fig
end