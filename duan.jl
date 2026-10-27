using CairoMakie, StructuredLight

include("analytical_correlations.jl")

Zs = LinRange(0, 2, 32)
ϵ = 0.1

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

ls = [(2, -1) (2, 0);
    (2, 1) (2, 3)]

with_theme(theme) do
    fig = Figure(size=(672, 403))

    for n ∈ eachindex(IndexCartesian(), ls)
        l₀, l₂ = ls[n]
        l₁ = 2l₀ - l₂
        pmax = l₀ * l₂ <= 0 ? 0 : min(2abs(l₀), abs(l₂))
        u = _lg(rs, rs, l=l₀)
        v2_profile = _lg(rs, rs, l=l₂)
        nseed = pmax + 2
        seed = (v2_profile + sum(
            _lg(rs, rs, l=l₁, w=1 / √3, p=p) for p in 0:pmax
        )) / √nseed
        u_seeded = sqrt(1 - abs2(ϵ)) * u + ϵ * seed
        v2 = v2_profile * √dA

        ax = Axis(fig[Tuple(n)...], xlabel=L"Z", ylabel=L"\text{Duan witness } D_{\text{opt}}", title=L"l_2 = %$l₂, \ p_{\max} = %$pmax")

        for p ∈ 0:3
            v1 = _lg(rs, rs, l=l₁, w=1 / √3, p=p) * √dA
            Ds = [duan(Z, u_seeded, v1, v2, NoDiffraction()) for Z in Zs]
            lines!(ax, Zs, Ds; label=L"p = %$p")
        end

        if Tuple(n) == size(ls)
            Legend(fig[:, end+1], ax)
        end
    end

    if !iszero(ϵ)
        Label(fig[0, :], L"\text{Simultaneous seeds: } \epsilon = %$ϵ,\quad c_2=c_{1,p}=1/\sqrt{p_{\max}+2}")
        save("Plots/duan_seeded.pdf", fig)
    else
        save("Plots/duan.pdf", fig)
    end
    fig
end
