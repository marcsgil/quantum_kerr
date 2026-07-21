using CairoMakie, StructuredLight, LinearAlgebra, EntanglementWitnesses, Clarabel, JuMP, Combinatorics

struct Linear end

function f(z, α, v1, v2, g, ::Linear)
    result = complex(zero(eltype(α)))
    for (a, b, c) in zip(α, v1, v2)
        result -= im * conj(b * c) * a^2 * g * z / 4
    end
    result
end

function h(z, α, v1, v2, g, ::Linear)
    complex(zero(eltype(α)))
end

struct NoDiffraction end

function f(z, α, v1, v2, g, ::NoDiffraction)
    Z = g * z / 4
    αz = α .* cis.(-Z * abs2.(α))

    (-im * Z * (v1 .* v2) ⋅ (αz .^ 2) - Z .^ 2 * (v1 .* v2 .* αz) ⋅ (αz .^ 3))
end

function h(z, α, v1, v2, g, ::NoDiffraction)
    Z = g * z / 4
    αz = α .* cis.(-Z * abs2.(α))
    Z .^ 2 * (v1 .* αz .^ 2) ⋅ (v2 .* αz .^ 2)
end

function γ(z, α, v1, v2, g, approximation)
    2 * real(v1 ⋅ v2 / 2 + f(z, α, v1, v2, g, approximation) + h(z, α, v1, v2, g, approximation))
end

function γ_matrix(z, α, vs, g, approximation)
    quadratures = reduce(vcat, [[v, -im * v] for v ∈ vs])
    [γ(z, α, v1, v2, g, approximation) for v1 ∈ quadratures, v2 ∈ quadratures]
end

function λ₋(z, α, v, g, approximation)
    1 / 2 + real(h(z, α, v, v, g, approximation)) + abs(f(z, α, v, v, g, approximation))
end

function duan(z, α, v1, v2, g, approximation)
    (1 + real(h(z, α, v1, v1, g, approximation) + h(z, α, v2, v2, g, approximation))
     -
     2 * abs(f(z, α, v1, v2, g, approximation)))
end

function partition2label(part)
    alphabet = collect('A':'Z')
    mapreduce(part -> prod(x->alphabet[x], part), (str1, str2) -> str1 * "|" * str2, part)
end
##
L = 8
N = 128
dr = L / N
rs = LinRange(-L / 2, L / 2 - dr, N)
α₀ = 5e4

g = 1e-7
dA = dr^2
##
zs = LinRange(0, 0.04, 16)

approximations = [NoDiffraction()]

# Duan

theme = merge(
    theme_latexfonts(),
    Theme(
        linewidth=4,
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
    fig = Figure(size=(1000, 600))

    for n ∈ eachindex(IndexCartesian(), ls)
        l₀, l₂ = ls[n]
        v0 = lg(rs, rs, l=l₀)
        α = α₀ * v0

        w1 = 1 / √3
        l₁ = 2l₀ - l₂

        ax = Axis(fig[Tuple(n)...], xlabel=L"z/z_R", ylabel="Duan Criterion", title=L"l_0 = %$l₀, l_2 = %$l₂")

        for p ∈ 0:3
            v2 = lg(rs, rs, l=l₂) * √dA
            v1 = lg(rs, rs, l=l₁, w=w1, p=p) * √dA
            for approximation ∈ approximations
                Ds = [duan(z, α, v1, v2, g, approximation) for z in zs]
                label = approximation == first(approximations) ? L"p = %$p" : nothing
                lines!(ax, zs, Ds; label)
            end
        end

        if Tuple(n) == size(ls)
            Legend(fig[:, end+1], ax)
        end
    end
    fig
end
##
zs = LinRange(0, 0.04, 16)
wits = similar(zs)

approximations = [NoDiffraction()]

# Optimal two modes

theme = merge(
    theme_latexfonts(),
    Theme(
        linewidth=4,
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
    fig = Figure(size=(1000, 600))

    for n ∈ eachindex(IndexCartesian(), ls)
        l₀, l₂ = ls[n]
        v0 = lg(rs, rs, l=l₀)
        α = α₀ * v0

        w1 = 1 / √3
        l₁ = 2l₀ - l₂

        ax = Axis(fig[Tuple(n)...], xlabel=L"z/z_R", ylabel="Duan Criterion", title=L"l_0 = %$l₀, l_2 = %$l₂")

        for p ∈ 0:3
            v2 = lg(rs, rs, l=l₂) * √dA
            v1 = lg(rs, rs, l=l₁, w=w1, p=p) * √dA
            vs = [v1, v2]
            display(γ_matrix(0, α, vs, g, NoDiffraction()))
            for approximation ∈ approximations
                Threads.@threads for n ∈ eachindex(zs, wits)
                    model = Model(Clarabel.Optimizer)
                    w, Z = getWitness(FullyWit(), model, γ_matrix(zs[n], α, vs, g, NoDiffraction()))
                    wits[n] = w
                end
                label = approximation == first(approximations) ? L"p = %$p" : nothing
                lines!(ax, zs, wits; label)
            end
        end

        if Tuple(n) == size(ls)
            Legend(fig[:, end+1], ax)
        end
    end
    fig
end
##
# Optimal N modes
zs = LinRange(0, 0.08, 32)
wits = similar(zs)

theme = merge(
    theme_latexfonts(),
    Theme(
        linewidth=4,
        palette=(
            color=Makie.to_colormap(:Set2_7),
        ),
    )
)

with_theme(theme) do
    fig = Figure()

    l₀ = 0
    l₂ = 1
    l₁ = 2l₀ - l₂

    v0 = lg(rs, rs, l=l₀)
    α = α₀ * v0

    vs = reduce(vcat, [[lg(rs, rs, l=l₂) * √dA], [lg(rs, rs; p, l=l₁, w=1/√3) * √dA for p ∈ 0:2]])


    ax = Axis(fig[1, 1], xlabel=L"z/z_R", ylabel="Optimal Witness")

    for m ∈ 2:2
        for part in partitions(1:length(vs), m)
            Threads.@threads for n ∈ eachindex(zs, wits)
                model = Model(Clarabel.Optimizer)
                set_attributes(model, "tol_feas" => 1e-4, "tol_gap_abs" => 1e-4)
                wits[n] = getWitness(FullyWit(), model, γ_matrix(zs[n], α, vs, g, NoDiffraction()), part)[1]
            end
            lines!(ax, zs, wits; label=partition2label(part))
        end
    end
    Legend(fig[:, end+1], ax)
    fig
end
##
##
# Optimal Multi
zs = LinRange(0, 0.06, 32)
wits = similar(zs)

theme = merge(
    theme_latexfonts(),
    Theme(
        linewidth=4,
        palette=(
            color=Makie.to_colormap(:Set2_7),
        ),
    )
)

with_theme(theme) do
    fig = Figure()

    l₀ = 0
    l₂ = 1
    l₁ = 2l₀ - l₂

    v0 = lg(rs, rs, l=l₀)
    α = α₀ * v0

    ax = Axis(fig[1, 1], xlabel=L"z/z_R", ylabel="Optimal Witness")

    for pmax ∈ 0:2
        vs = reduce(vcat, [[lg(rs, rs, l=l₂) * √dA], [lg(rs, rs; p, l=l₁, w=1/√3) * √dA for p ∈ 0:pmax]])
        Threads.@threads for n ∈ eachindex(zs, wits)
            model = Model(Clarabel.Optimizer)
            set_attributes(model, "tol_feas" => 1e-4, "tol_gap_abs" => 1e-4)
            wits[n] = getWitness(MultiWit(), model, γ_matrix(zs[n], α, vs, g, NoDiffraction()))[1]
        end
        lines!(ax, zs, wits; label=L"p_{\text{max}} = %$pmax")

    end
    Legend(fig[:, end+1], ax)
    fig
end