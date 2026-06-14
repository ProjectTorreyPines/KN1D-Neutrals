const ADAS_DIR = @__DIR__

function _adas_path(filename::String)::String
    return joinpath(ADAS_DIR, filename)
end

function read_adf11(filename)
    lines = readlines(filename)
    header_tokens = split(split(lines[1], "/")[1])

    iz0    = parse(Int, header_tokens[1])
    n_ne   = parse(Int, header_tokens[2])
    n_Te   = parse(Int, header_tokens[3])
    is1min = parse(Int, header_tokens[4])
    is1max = parse(Int, header_tokens[5])

    is_separator(line) = occursin("/", line)

    function parse_numbers(line::AbstractString)::Vector{Float64}
        return [parse(Float64, x) for x in split(line)]
    end

    sections = Vector{Float64}[]
    current = Float64[]
    for line in lines[2:end]
        stripped = strip(line)
        if is_separator(line)
            push!(sections, current)
            current = Float64[]
        elseif !isempty(stripped) && !startswith(stripped, repeat("-", 10))
            try
                append!(current, parse_numbers(line))
            catch
            end
        end
    end
    push!(sections, current)

    grid_values = sections[1]
    ne_log10 = grid_values[1:n_ne]
    Te_log10 = grid_values[n_ne+1 : n_ne+n_Te]

    ne = exp10.(ne_log10)
    Te = exp10.(Te_log10)

    blocks = Matrix{Float64}[]
    z1_vals = Int[]
    block_idx = 1
    for line in lines[2:end]
        uline = uppercase(line)
        if is_separator(line) && occursin("Z1=", uline)
            z1_str = strip(split(split(uline, "Z1=")[2], "/")[1])
            push!(z1_vals, parse(Int, z1_str))
            raw = sections[block_idx + 1]
            rate_log10 = permutedims(reshape(raw[1:n_ne*n_Te], n_Te, n_ne))
            push!(blocks, exp10.(rate_log10))
            block_idx += 1
        end
    end

    return (
        Te = Te,
        ne = ne,
        data = blocks,
        z1 = z1_vals,
        iz0 = iz0,
        n_ne = n_ne,
        n_Te = n_Te,
        is1min = is1min,
        is1max = is1max,
    )
end


struct ADF11Interpolator{S}
    spl::S
    ne_lo::Float64
    ne_hi::Float64
    Te_lo::Float64
    Te_hi::Float64
end

@inline function _clamped_logs(itp::ADF11Interpolator, Te_eV::Real, ne_cm3::Real)::Tuple{Float64,Float64}
    Te_eV > 0 || error("Te_eV must be positive")
    ne_cm3 > 0 || error("ne_cm3 must be positive")

    lTe = clamp(log10(Te_eV), itp.Te_lo, itp.Te_hi)
    lne = clamp(log10(ne_cm3), itp.ne_lo, itp.ne_hi)
    return lTe, lne
end

@inline function (itp::ADF11Interpolator)(Te_eV::Real, ne_cm3::Real)::Float64
    lTe, lne = _clamped_logs(itp, Te_eV, ne_cm3)
    return exp10(itp.spl(lne, lTe))
end

function (itp::ADF11Interpolator)(Te_eV::AbstractVector{<:Real}, ne_cm3::AbstractVector{<:Real})::Vector{Float64}
    length(Te_eV) == length(ne_cm3) || error("Te_eV and ne_cm3 must have the same length")
    out = Vector{Float64}(undef, length(Te_eV))
    return eval!(out, itp, Te_eV, ne_cm3)
end

function (itp::ADF11Interpolator)(Te_eV::Real, ne_cm3::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(ne_cm3))
    return eval!(out, itp, Te_eV, ne_cm3)
end

function (itp::ADF11Interpolator)(Te_eV::AbstractVector{<:Real}, ne_cm3::Real)::Vector{Float64}
    out = Vector{Float64}(undef, length(Te_eV))
    return eval!(out, itp, Te_eV, ne_cm3)
end

function eval!(out::Vector{Float64},
               itp::ADF11Interpolator,
               Te_eV::AbstractVector{<:Real},
               ne_cm3::AbstractVector{<:Real})::Vector{Float64}
    length(out) == length(Te_eV) == length(ne_cm3) || error("length mismatch")

    @inbounds for i in eachindex(out, Te_eV, ne_cm3)
        lTe, lne = _clamped_logs(itp, Te_eV[i], ne_cm3[i])
        out[i] = exp10(itp.spl(lne, lTe))
    end
    return out
end

function eval!(out::Vector{Float64},
               itp::ADF11Interpolator,
               Te_eV::Real,
               ne_cm3::AbstractVector{<:Real})::Vector{Float64}
    length(out) == length(ne_cm3) || error("length mismatch")

    Te_eV > 0 || error("Te_eV must be positive")
    lTe = clamp(log10(Te_eV), itp.Te_lo, itp.Te_hi)

    @inbounds for i in eachindex(out, ne_cm3)
        ne_cm3[i] > 0 || error("All ne_cm3 values must be positive")
        lne = clamp(log10(ne_cm3[i]), itp.ne_lo, itp.ne_hi)
        out[i] = exp10(itp.spl(lne, lTe))
    end
    return out
end

function eval!(out::Vector{Float64},
               itp::ADF11Interpolator,
               Te_eV::AbstractVector{<:Real},
               ne_cm3::Real)::Vector{Float64}
    length(out) == length(Te_eV) || error("length mismatch")

    ne_cm3 > 0 || error("ne_cm3 must be positive")
    lne = clamp(log10(ne_cm3), itp.ne_lo, itp.ne_hi)

    @inbounds for i in eachindex(out, Te_eV)
        Te_eV[i] > 0 || error("All Te_eV values must be positive")
        lTe = clamp(log10(Te_eV[i]), itp.Te_lo, itp.Te_hi)
        out[i] = exp10(itp.spl(lne, lTe))
    end
    return out
end

function make_adf11_interpolator(filename::String; block_index::Int=1)::ADF11Interpolator{Spline2D}
    d = read_adf11(filename)

    1 <= block_index <= length(d.data) || error("block_index=$block_index out of range")

    Te = d.Te
    ne = d.ne
    rates = d.data[block_index]

    log_Te = log10.(Te)
    log_ne = log10.(ne)
    log_rates = log10.(rates)

    spl = Spline2D(log_ne, log_Te, log_rates; kx=3, ky=3, s=0.0)

    return ADF11Interpolator(
        spl,
        log_ne[1], log_ne[end],
        log_Te[1], log_Te[end],
    )
end

# Build interpolators once at import time
const _scd_interp = make_adf11_interpolator(_adas_path("scd12_h.dat"); block_index=1)
const _acd_interp = make_adf11_interpolator(_adas_path("acd12_h.dat"); block_index=1)


function scd_adas(ne_m3::Real, Te_eV::Real)::Float64
    return _scd_interp(Te_eV, Float64(ne_m3) * 1.0e-6) * 1.0e-6
end

function scd_adas(ne_m3::AbstractVector{<:Real}, Te_eV::AbstractVector{<:Real})::Vector{Float64}
    return _scd_interp(Te_eV, Float64.(ne_m3) .* 1.0e-6) .* 1.0e-6
end

function scd_adas(ne_m3::Real, Te_eV::AbstractVector{<:Real})::Vector{Float64}
    return _scd_interp(Te_eV, Float64(ne_m3) * 1.0e-6) .* 1.0e-6
end

function scd_adas(ne_m3::AbstractVector{<:Real}, Te_eV::Real)::Vector{Float64}
    return _scd_interp(Te_eV, Float64.(ne_m3) .* 1.0e-6) .* 1.0e-6
end

function acd_adas(ne_m3::Real, Te_eV::Real)::Float64
    return _acd_interp(Te_eV, Float64(ne_m3) * 1.0e-6) * 1.0e-6
end

function acd_adas(ne_m3::AbstractVector{<:Real}, Te_eV::AbstractVector{<:Real})::Vector{Float64}
    return _acd_interp(Te_eV, Float64.(ne_m3) .* 1.0e-6) .* 1.0e-6
end

function acd_adas(ne_m3::Real, Te_eV::AbstractVector{<:Real})::Vector{Float64}
    return _acd_interp(Te_eV, Float64(ne_m3) * 1.0e-6) .* 1.0e-6
end

function acd_adas(ne_m3::AbstractVector{<:Real}, Te_eV::Real)::Vector{Float64}
    return _acd_interp(Te_eV, Float64.(ne_m3) .* 1.0e-6) .* 1.0e-6
end
