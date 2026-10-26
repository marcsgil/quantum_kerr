using CairoMakie, StructuredLight

include("analytical_correlations.jl")

Zs = LinRange(0, 2, 32)
ϵ = 0.0
nseed = 2
seed_coefficient = 1 / √nseed

theme = merge(
    theme_latexfonts(),
    Theme(
        fontsize=13.333333,
        linewidth=2,
        palette=(
            color=Makie.to_colormap(:Set2_5),
        ),
    )
)

ls = [(-1, 1) (0, 1);
    (1, 2) (1, 3)]

with_theme(theme) do
    fig = Figure(size=(672, 403))

    for n ∈ eachindex(IndexCartesian(), ls)
        l₀, l₂ = ls[n]
        l₁ = 2l₀ - l₂
        pmax = l₀ * l₂ <= 0 ? 0 : min(2abs(l₀), abs(l₂))
        u = _lg(rs, rs, l=l₀)

        ax = Axis(fig[Tuple(n)...], xlabel=L"Z", ylabel="Duan Criterion", title=L"l_0 = %$l₀, l_2 = %$l₂, p_{\max} = %$pmax")

        for p ∈ 0:3
            v2 = _lg(rs, rs, l=l₂) * √dA
            v1 = _lg(rs, rs, l=l₁, w=1 / √3, p=p) * √dA
            seed = seed_coefficient * (v1 + v2)
            u_seeded = (u + ϵ * seed) / √(1 + ϵ^2)
            Ds = [duan((1 + ϵ^2) * Zp, u_seeded, v1, v2, NoDiffraction()) for Zp in Zs]
            lines!(ax, Zs, Ds; label=L"p = %$p")
        end

        if Tuple(n) == size(ls)
            Legend(fig[:, end+1], ax)
        end
    end
    # Label(fig[0, :], L"\text{Two-mode seeds: } \epsilon = %$ϵ,\quad c_1=c_2=1/\sqrt{%$nseed}")
    save("Plots/duan.pdf", fig)
    fig
end
