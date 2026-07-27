using CairoMakie, StructuredLight

include("analytical_correlations.jl")

Zs = LinRange(0, 2, 32)

theme = merge(
    theme_latexfonts(),
    Theme(
        linewidth=4,
        palette=(
            color=Makie.to_colormap(:Set2_5),
        ),
    )
)

ls = [(-1, 1) (0, 1);
    (1, 2) (1, 3)]

with_theme(theme) do
    fig = Figure(size=(1000, 600))

    for n ∈ eachindex(IndexCartesian(), ls)
        l₀, l₂ = ls[n]
        l₁ = 2l₀ - l₂
        u = lg(rs, rs, l=l₀)

        ax = Axis(fig[Tuple(n)...], xlabel=L"Z", ylabel="Duan Criterion", title=L"l_0 = %$l₀, l_2 = %$l₂")

        for p ∈ 0:3
            v2 = lg(rs, rs, l=l₂) * √dA
            v1 = lg(rs, rs, l=l₁, w=1 / √3, p=p) * √dA
            Ds = [duan(Z, u, v1, v2, NoDiffraction()) for Z in Zs]
            lines!(ax, Zs, Ds; label=L"p = %$p")
        end

        if Tuple(n) == size(ls)
            Legend(fig[:, end+1], ax)
        end
    end
    save("Plots/duan.pdf", fig)
    fig
end