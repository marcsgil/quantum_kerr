using CairoMakie

# Draw a phase-coded mode map for the multimode-witness example:
# l₀ = 2, l₂ = 1, l₁ = 3, with pmax = 1.

const RGBf = Makie.RGBf
const RGBAf = Makie.RGBAf
const L = CairoMakie.Makie.LaTeXStrings.LaTeXString
const LATIN_MODERN = "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf"
set_theme!(font=LATIN_MODERN)

# Draw every profile on the same physical transverse scale, measured in units
# of the pump waist w. The mode-specific waist then sets its actual width.
function lg_mode(p::Int, l::Int; waist=1.0, n=161, extent=3.2)
    xs = range(-extent, extent; length=n)
    ys = range(-extent, extent; length=n)
    image = Matrix{RGBAf}(undef, n, n)
    ψs = Matrix{ComplexF64}(undef, n, n)
    abs_l = abs(l)
    factorial_ratio = prod(1:p) / prod(1:(p + abs_l))
    norm = sqrt(2 / π * factorial_ratio)

    for j in eachindex(ys), i in eachindex(xs)
        x, y = xs[i], ys[j]
        r = hypot(x, y)
        θ = atan(y, x)
        ξ = 2r^2 / waist^2
        # Generalized Laguerre polynomial L_p^|l|(ξ), evaluated explicitly.
        laguerre = sum((-1)^k * binomial(p + abs_l, p - k) * ξ^k / factorial(k)
                       for k in 0:p)
        ψ = norm * ξ^(abs_l / 2) * laguerre * exp(-ξ / 2) * cis(l * θ)
        ψs[j, i] = ψ
    end
    max_intensity = maximum(abs2, ψs)
    for j in eachindex(ys), i in eachindex(xs)
        ψ = ψs[j, i]
        intensity = abs2(ψ) / max_intensity
        α = 0.96 * sqrt(intensity) # transparency carries the local intensity
        image[j, i] = RGBAf(phase_color(angle(ψ))..., α)
    end
    image
end

function lg_radial_amplitude(r, p::Int, l::Int, waist::Real)
    abs_l = abs(l)
    ξ = 2r^2 / waist^2
    laguerre = sum((-1)^k * binomial(p + abs_l, p - k) * ξ^k / factorial(k)
                   for k in 0:p)
    sqrt(2 / π * prod(1:p) / prod(1:(p + abs_l))) *
        ξ^(abs_l / 2) * laguerre * exp(-ξ / 2)
end

# Radius at which the rendered profile has faded below the chosen normalized
# intensity. Links stop at this visible boundary instead of running to centers.
function visible_mode_radius(p::Int, l::Int, waist::Real, width::Real;
                             extent=3.2, threshold=0.015)
    radii = range(0, extent; length=2001)
    intensities = abs2.(lg_radial_amplitude.(radii, p, l, waist))
    cutoff = threshold * maximum(intensities)
    last_visible = findlast(>(cutoff), intensities)
    width / 2 * radii[last_visible] / extent
end

function phase_color(phase)
    # Cyclic phase map: blue → magenta → yellow → cyan → blue.
    stops = ((0.0, (0.20, 0.42, 0.78)),
             (0.25, (0.68, 0.28, 0.70)),
             (0.50, (0.96, 0.64, 0.22)),
             (0.75, (0.16, 0.68, 0.68)),
             (1.0, (0.20, 0.42, 0.78)))
    t = mod(phase / (2π) + 1, 1)
    for k in 1:length(stops)-1
        if stops[k][1] <= t <= stops[k+1][1]
            t0, c0 = stops[k]
            t1, c1 = stops[k+1]
            q = (t - t0) / (t1 - t0)
            return ntuple(j -> (1-q)*c0[j] + q*c1[j], 3)
        end
    end
    stops[end][2]
end

function wavy_link(a, b; start_offset=0.0, end_offset=0.0,
                   amplitude=1.0, cycles=2, n=180)
    dx0, dy0 = b[1]-a[1], b[2]-a[2]
    length0 = hypot(dx0, dy0)
    ux, uy = dx0/length0, dy0/length0
    a = (a[1] + start_offset*ux, a[2] + start_offset*uy)
    b = (b[1] - end_offset*ux, b[2] - end_offset*uy)
    dx, dy = b[1]-a[1], b[2]-a[2]
    length = hypot(dx, dy)
    nx, ny = -dy/length, dx/length
    t = range(0, 1; length=n)
    x = [a[1] + q*dx + amplitude*sin(2π*cycles*q)*nx for q in t]
    y = [a[2] + q*dy + amplitude*sin(2π*cycles*q)*ny for q in t]
    x, y
end

fig = Figure(size=(555, 240), backgroundcolor=:white)
X(x) = 2.2x # compensate for the wide figure while preserving circular mode profiles
ax = Axis(fig[1, 1], limits=(4, 226, 0, 96), aspect=DataAspect())
hidedecorations!(ax)
hidespines!(ax)

# A plain nonlinear-medium cell avoids implying a particular crystal structure.
# It is centered on the propagation arrow, with equal free shaft lengths either side.
poly!(ax, Point2f[(X(28.5),34), (X(37.5),34), (X(37.5),66), (X(28.5),66)];
      color=RGBf(0.96, 0.98, 0.99), strokecolor=RGBf(0.34, 0.43, 0.49), strokewidth=2.2)
text!(ax, X(33.0), 50; text=L"\chi^{(3)}", align=(:center, :center),
      fontsize=18, color=RGBf(0.18, 0.27, 0.33))

# Pump path through the medium and onward into the common output field.
lines!(ax, [X(20), X(28.5)], [50, 50]; color=RGBf(0.28, 0.38, 0.43), linewidth=2.2)
lines!(ax, [X(37.5), X(45.5)], [50, 50]; color=RGBf(0.28, 0.38, 0.43), linewidth=2.2)
poly!(ax, Point2f[(X(46),50), (X(44.4),51.2), (X(44.4),48.8)]; color=RGBf(0.28, 0.38, 0.43))

# A is the circumcenter; the three partners lie on its circumference and are
# symmetric about the horizontal axis shared with the propagation arrow.
const output_radius = 40.0 # measured in the plotted coordinates
const output_angle = deg2rad(45)
const input_mode_x = X(11.5)
const arrow_input_end_x = X(20.0)
const arrow_output_tip_x = X(46.0)
const input_gap = arrow_input_end_x - input_mode_x
const b0_center_x = arrow_output_tip_x + input_gap
const output_center_x = b0_center_x + output_radius
const output_shift = X(73.0) - output_center_x
const center_A = (output_center_x, 50.0)
centers = Dict(:pump => (X(11.5), 50.0), :A => center_A,
               :B0 => (center_A[1] - output_radius, center_A[2]),
               :B1 => (center_A[1] + output_radius*cos(output_angle),
                       center_A[2] + output_radius*sin(output_angle)),
               :C => (center_A[1] + output_radius*cos(output_angle),
                      center_A[2] - output_radius*sin(output_angle)))

const w = 1.0
const w2 = w
const w1 = inv(sqrt(2 / w^2 + 1 / w2^2))
mode_specs = ((:pump, 0, 2, 32, w), (:A, 0, 1, 32, w2),
              (:B0, 0, 3, 30, w1), (:B1, 1, 3, 30, w1),
              (:C, 2, 3, 30, w1))
link_radii = Dict(key => visible_mode_radius(p, l, waist, width)
                   for (key, p, l, width, waist) in mode_specs)

# Pairwise certification links stop at the visible edge of each mode profile.
for key in (:B0, :B1)
    x, y = wavy_link(centers[:A], centers[key]; start_offset=link_radii[:A],
                     end_offset=link_radii[key])
    lines!(ax, x, y; color=RGBf(0.18, 0.39, 0.52), linewidth=2.3)
end
x, y = wavy_link(centers[:A], centers[:C]; start_offset=link_radii[:A],
                 end_offset=link_radii[:C], amplitude=0.8, cycles=2)
lines!(ax, x, y; color=RGBf(0.45, 0.49, 0.51), linewidth=2.0, linestyle=:dot)

function mode_icon!(ax, center, p, l; width=26, waist=1.0)
    cx, cy = center
    half = width/2
    half_x = half
    image!(ax, (cx-half_x, cx+half_x), (cy-half, cy+half),
           lg_mode(p, l; waist);
           interpolate=true)
end

for (key, p, l, width, waist) in mode_specs
    cx, cy = centers[key]
    mode_icon!(ax, (cx, cy), p, l; width, waist)
end

text!(ax, X(11.5), 34.5; text="Input pump", align=(:center, :top),
      fontsize=14, color=RGBf(0.18, 0.27, 0.33))
text!(ax, X(11.5), 24.0; text=L"LG_{0,2}\ (l_0=2)", align=(:center, :top),
      fontsize=13, color=RGBf(0.18, 0.27, 0.33))
text!(ax, X(77.5) - output_shift, 50; text=L"\mathcal{A}: LG_{0,1}\ (l_2=1)", align=(:left, :center),
      fontsize=13.5, color=RGBf(0.18, 0.27, 0.33))
text!(ax, centers[:B0][1], 59; text=L"\mathcal{B}_0: LG_{0,3}", align=(:center, :bottom), fontsize=13,
      color=RGBf(0.18, 0.27, 0.33))
text!(ax, X(86.5) - output_shift + 10.5, centers[:B1][2]; text=L"\mathcal{B}_1: LG_{1,3}", align=(:left, :center), fontsize=13,
      color=RGBf(0.18, 0.27, 0.33))
text!(ax, X(86.5) - output_shift + 10.5, centers[:C][2]; text=L"\mathcal{C}: LG_{2,3}", align=(:left, :center), fontsize=13,
      color=RGBf(0.18, 0.27, 0.33))

# Compact visual key: solid means Duan-certified; dashed means inconclusive.
lines!(ax, [X(9), X(13)], [7, 7]; color=RGBf(0.18, 0.39, 0.52), linewidth=2.3)
text!(ax, X(16), 7; text="Duan-certified pair", align=(:left, :center), fontsize=12.5,
      color=RGBf(0.26, 0.33, 0.36))
lines!(ax, [X(47), X(51)], [7, 7]; color=RGBf(0.45, 0.49, 0.51), linewidth=2.0, linestyle=:dot)
text!(ax, X(53), 7; text="First-order inconclusive", align=(:left, :center), fontsize=12.5,
      color=RGBf(0.26, 0.33, 0.36))

save("Plots/radial_entanglement_schematic.pdf", fig)
fig
