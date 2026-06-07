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

g_eff = 4e-3
G = g_eff / (4 * dA)

U0 = (α * u0, conj(α * u0))
noise_prototype = similar.(U0, Float32)
param = (; G)

prob = GrossPitaevskiiProblem(U0, lengths; dispersion, nonlinearity, position_noise_func, noise_prototype, param)
alg = StrangSplitting()
tspan = (0, 8e-2)
nsaves = 128
dt = tspan[end] / 128
##
ts, sol = solve(prob, alg, tspan; dt, nsaves, save_start=false)

save_animation(Array(abs2.(sol[1])), "test.mp4")
##
u0_many = stack(u0 for _ ∈ 1:1024) |> CuArray
U0 = (α * u0_many, conj(α * u0_many))
noise_prototype = similar.(U0, Float64)
prob = GrossPitaevskiiProblem(U0, lengths; dispersion, nonlinearity, position_noise_func, noise_prototype, param)

v = cis(π / 4) * lg(rs, rs, l=0)
v1 = cis(π / 4) * lg(rs, rs, l=1)
v2 = cis(π / 4) * lg(rs, rs, l=-1)

v₊ = v1 + v2
v₋ = v1 - v2

V = hcat(vec.((v, v₊, v₋))...) * dA

rV = Reactant.to_rarray(V)
rU0 = Reactant.to_rarray.(U0)

f = @compile raw_observables(rU0..., rV)
##
ts, observables_vals = step_evolution(prob, tspan[end], f, rV; dt, nsaves=32)
ΔX², ΔP², duan = compose_raw(observables_vals)

R12 = imag.(sum(conj.(v1 .* v2) .* u0 .^ 2) * dA)

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=18, size=(1200,400))

    ax1 = Axis(fig[1,1], ylabel = "Quadrature Variance", xlabel=L"z/z_R")
    lines!(ax1, ts, ΔX², label = L"\Delta X^2", linewidth=4)
    lines!(ax1, ts, ΔP², label = L"\Delta P^2", linewidth=4)
    axislegend(ax1, position=:lt)

    ax2 = Axis(fig[1, 2], ylabel=L"\langle \Delta (X_1 \pm X_2) + \Delta (P_1 \mp P_2)\rangle", xlabel=L"z/z_R")
    lines!(ax2, ts, real.(duan), label="Positive P", linewidth=4)
    hlines!(ax2, [2], label="Duan bound", linestyle=:dash, color=:red, linewidth=4)
    lines!(ax2, ts, 2 .+ ts .* R12 * g_eff * α^2, label="Linear Theory", linestyle=:dot, linewidth=4, color=:green)
    axislegend(ax2, position=:lb)
    fig
end