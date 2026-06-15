using StructuredLight, CairoMakie, LinearAlgebra
include("observables.jl")

L = 8
lengths = (L, L)
N = 128
dr = L / N
rs = LinRange(-L / 2, L / 2 - dr, N)
α₀ = 5e4

α = α₀ * lg(rs, rs, l=0)

g = 1e-7
dA = dr^2

z_max = 0.05

zs = LinRange(0, z_max, 64)

@show g * z_max / 4dA
@show (g * z_max / 4)^2 * maximum(abs2, α) / dA

function f(z, v1, v2, α, g, dA)
    Z = g * z / 4
    αz = α .* cis.(-Z * abs2.(α))
    (-im * Z * (v1 .* v2) ⋅ (αz .^ 2) - Z .^ 2 * (v1 .* v2 .* αz) ⋅ (αz .^ 3)) * dA
end

function h(z, v1, v2, α, g, dA)
    Z = g * z / 4
    αz = α .* cis.(-Z * abs2.(α))
    Z .^ 2 * (v1 .* αz .^ 2) ⋅ (v2 .* αz .^ 2) * dA
end

function λ₋(z, v, α, g, dA)
    1 // 2 + real(h(z, v, v, α, g, dA)) - abs(f(z, v, v, α, g, dA))
end

function duan(z, v1, v2, α, g, dA)
    2 * (1 + real(h(z, v1, v1, α, g, dA) + h(z, v2, v2, α, g, dA))) - 4abs(f(z, v1, v2, α, g, dA))
end

function angle_squeezing(z, v, α, g, dA)
    select_angle(angle(f(z, v, v, α, g, dA)))
end

function angle_duan(z, v1, v2, α, g, dA)
    select_angle(angle(f(z, v1, v2, α, g, dA)))
end


v = lg(rs, rs, l=0)


λ₋s = map(z -> λ₋(z, v, α, g, dA), zs)
angles_squeezing = map(z -> angle_squeezing(z, v, α, g, dA), zs)



# R12 = abs((v1 .* v2) ⋅ (α .^ 2) * dA)
# R00 = abs((v .^ 2) ⋅ (α .^ 2) * dA)

# duans_linear = @. 2 - zs * abs(g * R12)
# λ₋_linear = @. 0.5 - zs * abs(g * R00) / 4

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=18, size=(1200, 700))

    ax1 = Axis(fig[1, 1], ylabel="Minimum Quadrature Variance (dB)")
    lines!(ax1, zs, decibels.(λ₋s), linewidth=4)
    # autolimits!(ax1)
    # lims = ax1.finallimits[]
    # lines!(ax1, zs, decibels.(λ₋_linear), label="Linear Theory", linestyle=:dot, linewidth=4, color=:black)
    # ylims!(ax1, lims.origin[2], lims.origin[2] + lims.widths[2])
    # axislegend(ax1, position=:rt)
    hidexdecorations!(ax1, ticks=false, grid=false)

    ax2 = Axis(fig[1, 2], ylabel=L"D_{\text{opt}} / D_0")
    ax4 = Axis(fig[2, 2], ylabel="Optimal Duan angle", xlabel=L"z/z_R", yticks=([-π / 2, -π / 4, 0, π / 4, π / 2], [L"-π/2", L"-π/4", L"0", L"π/4", L"π/2"]))
    # lines!(ax2, zs, real.(duans) / 2, label="Analytical Prediction", linewidth=4)
    # autolimits!(ax2)
    # lims = ax2.finallimits[]
    # lines!(ax2, zs, duans_linear / 2, label="Linear Theory", linestyle=:dot, linewidth=4, color=:black)
    # ylims!(ax2, lims.origin[2], lims.origin[2] + lims.widths[2])

    for l ∈ 1:6
        v1 = lg(rs, rs, l=l)
        v2 = lg(rs, rs, l=-l)

        duans = map(z -> duan(z, v1, v2, α, g, dA), zs)
        angles_duan = map(z -> angle_duan(z, v1, v2, α, g, dA), zs)

        lines!(ax2, zs, real.(duans) / 2, label=L"l = \pm %$l", linewidth=4)
        scatter!(ax4, zs, angles_duan)
    end

    # axislegend(ax2, position=:rt)
    Legend(fig[1, 3], ax2)
    hidexdecorations!(ax2, ticks=false, grid=false)

    ax3 = Axis(fig[2, 1], ylabel="Optimal squeezing angle", xlabel=L"z/z_R", yticks=([-π / 2, -π / 4, 0, π / 4, π / 2], [L"-π/2", L"-π/4", L"0", L"π/4", L"π/2"]))
    scatter!(ax3, zs, angles_squeezing)
    ylims!(ax3, -π / 2, π / 2)



    ylims!(ax4, -π / 2, π / 2)

    linkxaxes!(ax1, ax3)
    linkxaxes!(ax2, ax4)

    fig
end
##

zs = LinRange(0, 0.02, 64)
with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=18, size=(1000, 700))

    lmax = 3
    for l ∈ 0:lmax
        ax = Axis(fig[l ÷ 2, l % 2], xlabel = L"z/z_r", ylabel="VLF Criterion", title=L"l_1 = %$l")

        v1 = lg(rs, rs, l=l)

        for pmax ∈ 1:10
            v2 = sum(lg(rs, rs; p, l = -l) for p ∈ 1:pmax) / √pmax
            duans = map(z -> duan(z, v1, v2, α, g, dA), zs)
            lines!(ax, zs, real.(duans) / 2, label=L"p_{\text{max}} = %$pmax", linewidth=4)
        end

        if l == lmax
            Legend(fig[:, -1], ax)
        end
    end

    fig
end