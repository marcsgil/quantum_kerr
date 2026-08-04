using CairoMakie, EntanglementWitnesses, Clarabel, JuMP, Combinatorics, StructuredLight

include("analytical_correlations.jl")

zs = LinRange(0, 2, 16)
wits = similar(zs)

theme = merge(
    theme_latexfonts(),
    Theme(
        fontsize=13.333333,
        linewidth=3,
        palette=(
            color=Makie.to_colormap(:Set2_5),
            linestyle=[:solid]
        ),
        Lines=(cycle=[:color],)
    )
)

ls = [(-1, 1) (0, 1);
    (1, 2) (1, 3)]

with_theme(theme) do
    fig = Figure(size=(672, 403))

    for n ∈ eachindex(IndexCartesian(), ls)
        l₀, l₂ = ls[n]
        l₁ = 2l₀ - l₂
        u = _lg(rs, rs, l=l₀)

        ax = Axis(fig[Tuple(n)...], xlabel=L"Z", ylabel="Optimal Witness", title=L"l_0 = %$l₀, l_2 = %$l₂")
        Z = nothing

        for p ∈ 0:3
            v2 = _lg(rs, rs, l=l₂) * √dA
            v1 = _lg(rs, rs, l=l₁, w=1 / √3, p=p) * √dA
            vs = [v1, v2]
            Threads.@threads for n ∈ eachindex(zs, wits)
                model = Model(Clarabel.Optimizer)
                w, Z = getWitness(FullyWit(), model, γ_matrix(zs[n], u, vs, NoDiffraction()))
                wits[n] = w
            end
            lines!(ax, zs, wits; label=L"p = %$p")
        end

        if Tuple(n) == size(ls)
            Legend(fig[:, end+1], ax)
        end
    end
    save("Plots/optimal_two_modes.pdf", fig)
    fig
end
