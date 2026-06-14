struct KineticHConfig
    mesh_size::Int
    ion_rate::String
    ci_test::Bool
    alpha_cx_test::Bool
    grid_fctr::Float64
    extra_energy_bins_eV::Vector{Float64}
end

struct KineticH2Config
    mesh_size::Int
    grid_fctr::Float64
    extra_energy_bins_eV::Vector{Float64}
    ci_test::Bool
    alpha_cx_test::Bool
end

struct CollisionConfig
    H2_H_EL::Bool
    H2_H2_EL::Bool
    H2_P_EL::Bool
    H2_P_CX::Bool
    H_H_EL::Bool
    H_P_EL::Bool
    H_P_CX::Bool
    SIMPLE_CX::Bool
end

struct KN1DConfig
    kinetic_h::KineticHConfig
    kinetic_h2::KineticH2Config
    collisions::CollisionConfig
end

struct KN1DLinearInterp1D{TInterp}
    x::Vector{Float64}
    interpolant::TInterp
    fill_value::Float64
    extrapolate::Bool
end

struct KN1DLiteOutput{TXH, TFH, TNH, TGAMMA, TVXH, TTH, TQXH, TSION, TFHBC, TGAMMABC, TVR, TVX, TTNORM}
    xH::TXH
    fH::TFH
    nH::TNH
    GammaxH::TGAMMA
    VxH::TVXH
    TH::TTH
    qxH_total::TQXH
    Sion::TSION
    fHBC::TFHBC
    GammaxHBC::TGAMMABC
    vr::TVR
    vx::TVX
    Tnorm::TTNORM
end

struct Bound
    start::Int
    stop::Int

    function Bound(start::Int, stop::Int)
        start <= stop || throw(ArgumentError("start must be <= stop"))
        return new(start, stop)
    end
end

bound_slice(b::Bound) = b.start:b.stop


function get_local_directory(file_path::AbstractString)::String
    return dirname(realpath(file_path))
end

function get_json(file_path::AbstractString)
    open(file_path, "r") do io
        return JSON.parse(io)
    end
end

function get_config(config_path::AbstractString=DEFAULT_CONFIG_PATH)::KN1DConfig
    raw = get_json(config_path)
    return KN1DConfig(
        _parse_kinetic_h_config(raw["kinetic_h"]),
        _parse_kinetic_h2_config(raw["kinetic_h2"]),
        _parse_collision_config(raw["collisions"]),
    )
end

function debrief(statement, condition::Bool)::Nothing
    condition && println(statement)
    return nothing
end

function sval(value; length::Union{Nothing,Int}=nothing)::String
    text = strip(string(value))
    if isnothing(length)
        return text
    end
    return first(text, min(length, lastindex(text)))
end

function poly(x::Number, coeffs::AbstractVector{<:Real})::Float64
    n = length(coeffs)
    n == 0 && throw(ArgumentError("coeffs must not be empty"))

    xf = Float64(x)
    y = Float64(coeffs[n])
    @inbounds for i in (n - 1):-1:1
        y = muladd(y, xf, Float64(coeffs[i]))
    end
    return y
end

function poly(x::AbstractVector{<:Real}, coeffs::AbstractVector{<:Real})::Vector{Float64}
    n = length(coeffs)
    n == 0 && throw(ArgumentError("coeffs must not be empty"))

    y = fill(Float64(coeffs[n]), length(x))
    @inbounds for ci in (n - 1):-1:1
        c = Float64(coeffs[ci])
        for i in eachindex(x, y)
            y[i] = muladd(y[i], Float64(x[i]), c)
        end
    end
    return y
end

const INTERPOLATION_ENTRY_POINTS = (
    :linear_interp_1d,
    :interp_1d,
    :path_interp_2d,
    :bs2dr,
)

"""
Build a 1D linear interpolation object for KN1D profile data.

Method:
- linear interpolation on a gridded 1D mesh via `Interpolations.jl`

Extrapolation behavior:
- `fill_value="extrapolate"` uses linear extrapolation outside the tabulated domain
- numeric `fill_value` returns that constant outside the tabulated domain

Grid assumptions:
- when `assume_sorted=true`, `x` must already be sorted in ascending order
- when `assume_sorted=false`, `x` and `y` are sorted together to match
  `scipy.interpolate.interp1d(..., assume_sorted=False)`
"""
function linear_interp_1d(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real};
    kind::AbstractString="linear",
    axis::Int=1,
    fill_value=NaN,
    assume_sorted::Bool=false,
    )::KN1DLinearInterp1D
    x_sorted, y_sorted = _prepare_xy(x, y, kind, axis, assume_sorted)
    return _build_linear_interp_1d(x_sorted, y_sorted, fill_value)
end

function interp_1d(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    x_new::Real;
    kind::AbstractString="linear",
    axis::Int=1,
    fill_value=NaN,
    assume_sorted::Bool=false,
)::Float64
    return linear_interp_1d(x, y; kind=kind, axis=axis, fill_value=fill_value, assume_sorted=assume_sorted)(Float64(x_new))
end

function interp_1d(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    x_new::AbstractVector{<:Real};
    kind::AbstractString="linear",
    axis::Int=1,
    fill_value=NaN,
    assume_sorted::Bool=false,
)::Vector{Float64}
    itp = linear_interp_1d(x, y; kind=kind, axis=axis, fill_value=fill_value, assume_sorted=assume_sorted)
    out = Vector{Float64}(undef, length(x_new))
    @inbounds for i in eachindex(x_new, out)
        out[i] = itp(Float64(x_new[i]))
    end
    return out
end

@inline function (itp::KN1DLinearInterp1D)(x_new::Float64)::Float64
    if itp.extrapolate
        return itp.interpolant(x_new)
    end

    if x_new < itp.x[1] || x_new > itp.x[end]
        return itp.fill_value
    end

    return itp.interpolant(x_new)
end

function evaluate!(out::Vector{Float64}, itp::KN1DLinearInterp1D, x_new::AbstractVector{<:Real})::Vector{Float64}
    length(out) == length(x_new) || throw(DimensionMismatch("out and x_new must have the same length"))
    @inbounds for i in eachindex(out, x_new)
        out[i] = itp(Float64(x_new[i]))
    end
    return out
end

function path_interp_2d(
    p::AbstractMatrix{<:Real},
    px::AbstractVector{<:Real},
    py::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)::Vector{Float64}
    length(px) == size(p, 1) || throw(DimensionMismatch("length(px) must match size(p, 1)"))
    length(py) == size(p, 2) || throw(DimensionMismatch("length(py) must match size(p, 2)"))
    length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    issorted(px) || throw(ArgumentError("px must be sorted ascending"))
    issorted(py) || throw(ArgumentError("py must be sorted ascending"))

    out = Vector{Float64}(undef, length(x))
    p_f = Float64.(p)
    px_f = Float64.(px)
    py_f = Float64.(py)

    @inbounds for n in eachindex(x, y, out)
        xv = Float64(x[n])
        yv = Float64(y[n])
        if xv < px_f[1] || xv > px_f[end] || yv < py_f[1] || yv > py_f[end]
            throw(BoundsError((px_f, py_f), (xv, yv)))
        end

        ix = searchsortedlast(px_f, xv)
        iy = searchsortedlast(py_f, yv)
        ix == length(px_f) && (ix -= 1)
        iy == length(py_f) && (iy -= 1)

        x0 = px_f[ix]
        x1 = px_f[ix + 1]
        y0 = py_f[iy]
        y1 = py_f[iy + 1]
        tx = (xv - x0) / (x1 - x0)
        ty = (yv - y0) / (y1 - y0)

        p00 = p_f[ix, iy]
        p10 = p_f[ix + 1, iy]
        p01 = p_f[ix, iy + 1]
        p11 = p_f[ix + 1, iy + 1]
        out[n] = (1.0 - tx) * (1.0 - ty) * p00 +
                 tx * (1.0 - ty) * p10 +
                 (1.0 - tx) * ty * p01 +
                 tx * ty * p11
    end
    return out
end

function bs2dr(Spl::Spline2D, x::Vector{Float64}, y::Vector{Float64})::Vector{Float64}
    return Dierckx.evaluate(Spl, x, y)
end

function reverse_dim(a::AbstractArray, dim::Int)::typeof(a)
    1 <= dim <= ndims(a) || throw(ArgumentError("dim must satisfy 1 <= dim <= ndims(a)"))
    return reverse(a; dims=dim)
end

const KN1D_LITE_OUTPUT_KEYS = (
    :xH,
    :fH,
    :nH,
    :GammaxH,
    :VxH,
    :TH,
    :qxH_total,
    :Sion,
    :fHBC,
    :GammaxHBC,
    :vr,
    :vx,
    :Tnorm,
)

const KN1D_OUTPUT_FILES = (
    :KN1D_input,
    :KN1D_mesh,
    :KN1D_H2,
    :KN1D_H,
)

const KN1D_OUTPUT_SCHEMA = (
    KN1D_input = (
        :x, :xlimiter, :xsep, :GaugeH2, :mu, :Ti, :Te, :n, :vxi, :LC, :PipeDia, :truncate,
        :xH2, :TiM, :TeM, :nM, :PipeDiaM, :vxM, :vrM, :TnormM,
        :xH, :TiA, :TeA, :nA, :PipeDiaA, :vxA, :vrA, :TnormA,
    ),
    KN1D_mesh = (
        :x_s, :GaugeH2_s, :mu_s, :Ti_s, :Te_s, :n_s, :vxi_s, :LC_s, :PipeDia_s,
        :xH2_s, :vxM_s, :vrM_s, :TnormM_s,
        :xH_s, :vxA_s, :vrA_s, :TnormA_s,
    ),
    KN1D_H2 = (
        :xH2, :fH2, :nH2, :GammaxH2, :VxH2, :pH2, :TH2, :qxH2, :qxH2_total, :Sloss,
        :QH2, :RxH2, :QH2_total, :AlbedoH2, :nHP, :THP, :fSH, :SH, :SP, :SHP, :NuE,
        :NuDis, :piH2_xx, :piH2_yy, :piH2_zz, :RxH2CX, :RxH_H2, :RxP_H2, :RxW_H2,
        :EH2CX, :EH_H2, :EP_H2, :EW_H2, :Epara_PerpH2_H2, :GammaxH2_plus, :GammaxH2_minus,
    ),
    KN1D_H = (
        :xH, :fH, :nH, :GammaxH, :VxH, :pH, :TH, :qxH, :qxH_total, :NetHSource, :Sion,
        :SideWallH, :QH, :RxH, :QH_total, :AlbedoH, :GammaHLim, :piH_xx, :piH_yy, :piH_zz,
        :RxHCX, :RxH2_H, :RxP_H, :RxW_H, :EHCX, :EH2_H, :EP_H, :EW_H, :Epara_PerpH_H,
        :SourceH, :SRecomb, :EH_hist, :SI_hist, :Lyman, :Balmer,
    ),
)

function kn1d_lite_output_placeholder()::KN1DLiteOutput{Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing}
    return KN1DLiteOutput(
        nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing,
    )
end

function kn1d_output_placeholders()
    return (
        KN1D_input = NamedTuple{KN1D_OUTPUT_SCHEMA.KN1D_input}(ntuple(_ -> nothing, length(KN1D_OUTPUT_SCHEMA.KN1D_input))),
        KN1D_mesh = NamedTuple{KN1D_OUTPUT_SCHEMA.KN1D_mesh}(ntuple(_ -> nothing, length(KN1D_OUTPUT_SCHEMA.KN1D_mesh))),
        KN1D_H2 = NamedTuple{KN1D_OUTPUT_SCHEMA.KN1D_H2}(ntuple(_ -> nothing, length(KN1D_OUTPUT_SCHEMA.KN1D_H2))),
        KN1D_H = NamedTuple{KN1D_OUTPUT_SCHEMA.KN1D_H}(ntuple(_ -> nothing, length(KN1D_OUTPUT_SCHEMA.KN1D_H))),
    )
end

function _parse_kinetic_h_config(raw)::KineticHConfig
    return KineticHConfig(
        Int(raw["mesh_size"]),
        String(raw["ion_rate"]),
        Bool(raw["ci_test"]),
        Bool(raw["alpha_cx_test"]),
        Float64(raw["grid_fctr"]),
        _float_vector(get(raw, "extra_energy_bins_eV", Float64[])),
    )
end

function _parse_kinetic_h2_config(raw)::KineticH2Config
    return KineticH2Config(
        Int(raw["mesh_size"]),
        Float64(raw["grid_fctr"]),
        _float_vector(get(raw, "extra_energy_bins_eV", Float64[])),
        Bool(raw["ci_test"]),
        Bool(raw["alpha_cx_test"]),
    )
end

function _parse_collision_config(raw)::CollisionConfig
    return CollisionConfig(
        Bool(raw["H2_H_EL"]),
        Bool(raw["H2_H2_EL"]),
        Bool(raw["H2_P_EL"]),
        Bool(raw["H2_P_CX"]),
        Bool(raw["H_H_EL"]),
        Bool(raw["H_P_EL"]),
        Bool(raw["H_P_CX"]),
        Bool(raw["SIMPLE_CX"]),
    )
end

function _float_vector(values)::Vector{Float64}
    out = Vector{Float64}(undef, length(values))
    @inbounds for i in eachindex(values, out)
        out[i] = Float64(values[i])
    end
    return out
end

function _prepare_xy(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    kind::AbstractString,
    axis::Int,
    assume_sorted::Bool,
)::Tuple{Vector{Float64}, Vector{Float64}}
    kind == "linear" || throw(ArgumentError("interp_1d currently supports only kind=\"linear\""))
    axis == 1 || throw(ArgumentError("interp_1d currently supports only axis=1 for Julia vectors"))
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length"))
    length(x) >= 2 || throw(ArgumentError("interp_1d requires at least two points"))

    x_vals = _float_vector(x)
    y_vals = _float_vector(y)

    if assume_sorted
        issorted(x_vals) || throw(ArgumentError("x must be sorted when assume_sorted=true"))
        return x_vals, y_vals
    end

    if issorted(x_vals)
        return x_vals, y_vals
    end

    perm = sortperm(x_vals)
    return x_vals[perm], y_vals[perm]
end

function _build_linear_interp_1d(
    x_sorted::Vector{Float64},
    y_sorted::Vector{Float64},
    fill_value::Real,
)::KN1DLinearInterp1D
    base = interpolate((x_sorted,), y_sorted, Gridded(Linear()))
    return KN1DLinearInterp1D(x_sorted, base, Float64(fill_value), false)
end

function _build_linear_interp_1d(
    x_sorted::Vector{Float64},
    y_sorted::Vector{Float64},
    fill_value::AbstractString,
)::KN1DLinearInterp1D
    fill_value == "extrapolate" || throw(ArgumentError("unsupported fill_value string: $fill_value"))
    base = interpolate((x_sorted,), y_sorted, Gridded(Linear()))
    ext = extrapolate(base, Line())
    return KN1DLinearInterp1D(x_sorted, ext, 0.0, true)
end
