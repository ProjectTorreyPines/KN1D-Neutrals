function sigmav_rec_h1s(Te::Real)::Float64
    Tec = clamp(Float64(Te), 0.1, 2.01e4)

    n = 1.0
    Ry = 13.58
    Eion_n = Ry / n
    Anl = 3.92
    Xnl = 0.35

    Bn = Eion_n / Tec
    return Anl * 1e-14 * sqrt(Eion_n / Ry) * ((Bn^1.5) / (Bn + Xnl)) * 1e-6
end

function sigmav_rec_h1s(Te::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(Te))
    @inbounds for i in eachindex(Te, out)
        out[i] = sigmav_rec_h1s(Te[i])
    end
    return out
end
