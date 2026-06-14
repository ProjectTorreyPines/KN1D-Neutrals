struct JohnsonHinnov
    dknot::Vector{Float64}
    tknot::Vector{Float64}
    order::Int
    logr_bscoef::Array{Float64, 3}
    logs_bscoef::Vector{Float64}
    logalpha_bscoef::Vector{Float64}
    a_lyman::Vector{Float64}
    a_balmer::Vector{Float64}
    r_splines::Matrix{Spline2D}
    s_spline::Spline2D
    alpha_spline::Spline2D
end


@inline _jh_data_path() = joinpath(@__DIR__, "jh_bscoef.npz")

function _jh_spline_from_coef(
    dknot::Vector{Float64},
    tknot::Vector{Float64},
    order::Int,
    coef::Vector{Float64},
    )::Spline2D
    degree = order - 1
    degree >= 1 || throw(ArgumentError("Johnson-Hinnov spline order must be >= 2"))
    ncoef_expected = (length(tknot) - degree - 1) * (length(dknot) - degree - 1)
    length(coef) == ncoef_expected || throw(DimensionMismatch(
        "Coefficient length $(length(coef)) does not match expected $ncoef_expected " *
        "for knots (tknot=$(length(tknot)), dknot=$(length(dknot))) and degree=$degree"
    ))
    # This orientation reproduces KN1DPy/SciPy results for the bundled coefficients.
    return Spline2D(dknot, tknot, coef, degree, degree, 0.0)
end

function JohnsonHinnov(; create::Bool=false)::JohnsonHinnov
    create && throw(ArgumentError("create=true is not implemented in KN1DJl; use bundled jh_bscoef.npz"))
    data = npzread(_jh_data_path())

    dknot = Float64.(vec(data["dknot"]))
    tknot = Float64.(vec(data["tknot"]))
    order = Int(data["order"])
    logr_bscoef = Float64.(Array(data["logr_bscoef"]))
    logs_bscoef = Float64.(vec(data["logs_bscoef"]))
    logalpha_bscoef = Float64.(vec(data["logalpha_bscoef"]))
    a_lyman = Float64.(vec(data["a_lyman"]))
    a_balmer = Float64.(vec(data["a_balmer"]))

    r_splines = Matrix{Spline2D}(undef, 2, 5)
    for ion in 0:1
        for p in 2:6
            # Match Python directly: self.logr_bscoef.T[:, Ion, p-2]
            coef = vec(PermutedDimsArray(logr_bscoef, (3, 2, 1))[:, ion + 1, p - 1])
            r_splines[ion + 1, p - 1] = _jh_spline_from_coef(dknot, tknot, order, coef)
        end
    end

    s_spline = _jh_spline_from_coef(dknot, tknot, order, logs_bscoef)
    alpha_spline = _jh_spline_from_coef(dknot, tknot, order, logalpha_bscoef)

    return JohnsonHinnov(
        dknot,
        tknot,
        order,
        logr_bscoef,
        logs_bscoef,
        logalpha_bscoef,
        a_lyman,
        a_balmer,
        r_splines,
        s_spline,
        alpha_spline,
    )
end

function jhr_spline(jh::JohnsonHinnov, ion::Int, p::Int)::Spline2D
    ion ∈ (0, 1) || throw(ArgumentError("ion must be 0 or 1"))
    2 <= p <= 6 || throw(ArgumentError("p must be in the range 2:6"))
    return jh.r_splines[ion + 1, p - 1]
end

function bs2dr_jh(spl::Spline2D, x::Vector{Float64}, y::Vector{Float64})::Vector{Float64}
    length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    return Dierckx.evaluate(spl, x, y)
end

function bs2dr_jh(spl::Spline2D, x::Float64, y::Float64)::Float64
    return Dierckx.evaluate(spl, Float64[x], Float64[y])[1]
end

function _log_inputs(ne::AbstractVector{<:Real}, Te::AbstractVector{<:Real})::Tuple{Vector{Float64}, Vector{Float64}}
    n = length(ne)
    length(Te) == n || throw(DimensionMismatch("ne and Te must have the same length"))
    lne = Vector{Float64}(undef, n)
    lte = Vector{Float64}(undef, n)
    @inbounds for i in eachindex(ne, Te, lne, lte)
        lne[i] = log(Float64(ne[i]))
        lte[i] = log(Float64(Te[i]))
    end
    return lne, lte
end

function _in_bounds_indices(
    lne::Vector{Float64},
    lte::Vector{Float64},
    dkmin::Float64,
    dkmax::Float64,
    tkmin::Float64,
    tkmax::Float64,
)::Vector{Int}
    ok = Int[]
    sizehint!(ok, length(lne))
    @inbounds for i in eachindex(lne, lte)
        if dkmin <= lne[i] <= dkmax && tkmin <= lte[i] <= tkmax
            push!(ok, i)
        end
    end
    return ok
end

function _coef_eval(
    spl::Spline2D,
    ne::AbstractVector{<:Real},
    Te::AbstractVector{<:Real},
    dknot::Vector{Float64},
    tknot::Vector{Float64},
    no_null::Bool,
)::Vector{Float64}
    n = length(ne)
    out = fill(1.0e32, n)
    n == 0 && return out

    lne, lte = _log_inputs(ne, Te)
    dkmin, dkmax = minimum(dknot), maximum(dknot)
    tkmin, tkmax = minimum(tknot), maximum(tknot)

    if no_null
        @inbounds for i in eachindex(lne, lte)
            lne[i] = clamp(lne[i], dkmin, dkmax)
            lte[i] = clamp(lte[i], tkmin, tkmax)
        end
        vals = bs2dr_jh(spl, lne, lte)
        @inbounds for i in eachindex(vals, out)
            out[i] = exp(vals[i])
        end
        return out
    end

    ok = _in_bounds_indices(lne, lte, dkmin, dkmax, tkmin, tkmax)
    isempty(ok) && return out

    lne_ok = Vector{Float64}(undef, length(ok))
    lte_ok = Vector{Float64}(undef, length(ok))
    @inbounds for j in eachindex(ok)
        i = ok[j]
        lne_ok[j] = lne[i]
        lte_ok[j] = lte[i]
    end

    vals = bs2dr_jh(spl, lne_ok, lte_ok)
    @inbounds for j in eachindex(ok, vals)
        out[ok[j]] = exp(vals[j])
    end
    return out
end

function _coef_eval(
    spl::Spline2D,
    ne::Real,
    Te::Real,
    dknot::Vector{Float64},
    tknot::Vector{Float64},
    no_null::Bool,
)::Float64
    lne = log(Float64(ne))
    lte = log(Float64(Te))
    dkmin, dkmax = minimum(dknot), maximum(dknot)
    tkmin, tkmax = minimum(tknot), maximum(tknot)

    if no_null
        return exp(bs2dr_jh(spl, clamp(lne, dkmin, dkmax), clamp(lte, tkmin, tkmax)))
    end

    if !(dkmin <= lne <= dkmax && tkmin <= lte <= tkmax)
        return 1.0e32
    end
    return exp(bs2dr_jh(spl, lne, lte))
end

function jhr_coef(
    jh::JohnsonHinnov,
    ne::AbstractVector{<:Real},
    Te::AbstractVector{<:Real},
    ion::Int,
    p::Int;
    no_null::Bool=false,
)::Vector{Float64}
    return _coef_eval(jhr_spline(jh, ion, p), ne, Te, jh.dknot, jh.tknot, no_null)
end

function jhs_coef(
    jh::JohnsonHinnov,
    density::Float64,
    Te::Float64;
    no_null::Bool = false,
)::Float64

    density > 0.0 || throw(ArgumentError("density must be positive"))
    Te > 0.0 || throw(ArgumentError("Te must be positive"))

    LD = log(density)
    LT = log(Te)

    dmin = minimum(jh.dknot)
    dmax = maximum(jh.dknot)
    tmin = minimum(jh.tknot)
    tmax = maximum(jh.tknot)

    if no_null
        LD = clamp(LD, dmin, dmax)
        LT = clamp(LT, tmin, tmax)
    else
        if !(dmin <= LD <= dmax && tmin <= LT <= tmax)
            return 1.0e32
        end
    end

    return exp(bs2dr_jh(jh.s_spline, LD, LT))
end

function jhs_coef(
    jh::JohnsonHinnov,
    density::AbstractVector{<:Real},
    Te::AbstractVector{<:Real};
    no_null::Bool = false,
)::Vector{Float64}

    length(density) == length(Te) ||
        throw(DimensionMismatch("density and Te must have the same length"))

    out = Vector{Float64}(undef, length(density))

    @inbounds for i in eachindex(density, Te)
        out[i] = jhs_coef(
            jh,
            Float64(density[i]),
            Float64(Te[i]);
            no_null = no_null,
        )
    end

    return out
end


function jhalpha_coef(
    jh::JohnsonHinnov,
    ne::Real,
    Te::Real;
    no_null::Bool=false,
)::Float64
    return _coef_eval(jh.alpha_spline, ne, Te, jh.dknot, jh.tknot, no_null)
end

function jhalpha_coef(
    jh::JohnsonHinnov,
    ne::AbstractVector{<:Real},
    Te::AbstractVector{<:Real};
    no_null::Bool=false,
)::Vector{Float64}
    return _coef_eval(jh.alpha_spline, ne, Te, jh.dknot, jh.tknot, no_null)
end

function nh_saha(jh::JohnsonHinnov, ne::AbstractVector{<:Real}, Te::AbstractVector{<:Real}, p::Int)::Vector{Float64}
    length(ne) == length(Te) || throw(DimensionMismatch("ne and Te must have the same length"))
    0 < p || throw(ArgumentError("p must be positive"))
    n = length(ne)
    result = fill(1.0e32, n)
    n == 0 && return result
    ok = findall((0.0 .< ne) .& (ne .< 1.0e32) .& (0.0 .< Te) .& (Te .< 1.0e32))
    isempty(ok) && return result
    result[ok] .= 3.310E-28 .* ((ne[ok] .* p) .^ 2) .* exp.(13.6057 ./ ((p .^ 2) .* Te[ok])) ./ (Te[ok] .^ 1.5)
    return result
end
        
function lyman_alpha(jh::JohnsonHinnov, ne::AbstractVector{<:Real}, Te::AbstractVector{<:Real}, N0::AbstractVector{<:Real}; no_null::Bool=false)::Vector{Float64}
    length(ne) == length(Te) || throw(DimensionMismatch("ne and Te must have the same length"))
    length(ne) == length(N0) || throw(DimensionMismatch("ne and N0 must have the same length"))
    n = length(ne)
    result = fill(1.0e32, n)
    photons = fill(1.0e32, n)
    r02 = jhr_coef(jh, ne, Te, 0, 2; no_null=no_null)
    r12 = jhr_coef(jh, ne, Te, 1, 2; no_null=no_null)
    NHSaha1 = nh_saha(jh, ne, Te, 1)
    NHSaha2 = nh_saha(jh, ne, Te, 2)
    ok = findall((0.0 .< N0) .& (N0 .< 1.0e32) .& (r02 .< 1.0e32) .& (r12 .< 1.0e32) .& (NHSaha1 .< 1.0e32) .& (NHSaha2 .< 1.0e32))
    photons[ok] = jh.a_lyman[1] .* (r02[ok] .+ (r12[ok] .* N0[ok] ./ NHSaha1[ok])) .* NHSaha2[ok]
    result[ok] = 13.6057 * 0.75 .* photons[ok] * 1.6e-19
    return result
end

function balmer_alpha(jh::JohnsonHinnov, ne::AbstractVector{<:Real}, Te::AbstractVector{<:Real}, N0::AbstractVector{<:Real}; no_null::Bool=false)::Vector{Float64}
    length(ne) == length(Te) || throw(DimensionMismatch("ne and Te must have the same length"))
    length(ne) == length(N0) || throw(DimensionMismatch("ne and N0 must have the same length"))
    n = length(ne)
    result = fill(1.0e32, n)
    photons = fill(1.0e32, n)
    r03 = jhr_coef(jh, ne, Te, 0, 3; no_null=no_null)
    r13 = jhr_coef(jh, ne, Te, 1, 3; no_null=no_null)
    NHSaha1 = nh_saha(jh, ne, Te, 1)
    NHSaha3 = nh_saha(jh, ne, Te, 3)
    ok = findall((0.0 .< N0) .& (N0 .< 1.0e32) .& (r03 .< 1.0e32) .& (r13 .< 1.0e32) .& (NHSaha1 .< 1.0e32) .& (NHSaha3 .< 1.0e32))
    photons[ok] = jh.a_balmer[1] .* (r03[ok] .+ (r13[ok] .* N0[ok] ./ NHSaha1[ok])) .* NHSaha3[ok]
    result[ok] = 13.6057 * (0.25 - 1.0/9.0) .* photons[ok] * 1.6e-19
    return result
end
