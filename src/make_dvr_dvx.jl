struct VSpaceDifferentials
    dvr_vol::Vector{Float64}
    dvr_vol_h_order::Vector{Float64}
    dvx::Vector{Float64}
    dvr::Vector{Float64}
    vr_left_bound::Vector{Float64}
    vr_right_bound::Vector{Float64}
    vx_left_bound::Vector{Float64}
    vx_right_bound::Vector{Float64}
    volume::Matrix{Float64}
    vth_dvx::Matrix{Float64}
    vx_dvx::Matrix{Float64}
    vr_dvr::Matrix{Float64}
    vmag_squared::Matrix{Float64}
    vx_pos_start::Int
    vx_pos_end::Int
    vx_neg_start::Int
    vx_neg_end::Int
end

function VSpaceDifferentials(vr_in::AbstractVector{<:Real}, vx_in::AbstractVector{<:Real})::VSpaceDifferentials
    vr = Float64.(vr_in)
    vx = Float64.(vx_in)
    nvr = length(vr)
    nvx = length(vx)
    nvr >= 2 || throw(ArgumentError("vr must contain at least two values"))
    nvx >= 2 || throw(ArgumentError("vx must contain at least two values"))

    # --- Calculations for r-dimension ---
    vr_left_bound = Vector{Float64}(undef, nvr)
    vr_right_bound = Vector{Float64}(undef, nvr)
    dvr = Vector{Float64}(undef, nvr)
    dvr_vol = Vector{Float64}(undef, nvr)
    dvr_vol_h_order = Vector{Float64}(undef, nvr)

    vr_left_bound[1] = 0.0
    @inbounds for i in 1:nvr
        vr_right = if i == nvr
            0.5 * (vr[i] + (2.0 * vr[i] - vr[i - 1]))
        else
            0.5 * (vr[i] + vr[i + 1])
        end
        vr_right_bound[i] = vr_right
        if i > 1
            vr_left_bound[i] = vr_right_bound[i - 1]
        end
        dvr[i] = vr_right_bound[i] - vr_left_bound[i]
        dvr_vol[i] = π * (vr_right_bound[i]^2 - vr_left_bound[i]^2)
        dvr_vol_h_order[i] = (4.0 / 3.0) * π * (vr_right_bound[i]^3 - vr_left_bound[i]^3)
    end

    # --- Calculations for x-dimension ---
    vx_left_bound = Vector{Float64}(undef, nvx)
    vx_right_bound = Vector{Float64}(undef, nvx)
    dvx = Vector{Float64}(undef, nvx)

    @inbounds for j in 1:nvx
        vx_left = if j == 1
            0.5 * ((2.0 * vx[1] - vx[2]) + vx[1])
        else
            0.5 * (vx[j - 1] + vx[j])
        end
        vx_right = if j == nvx
            0.5 * (vx[j] + (2.0 * vx[j] - vx[j - 1]))
        else
            0.5 * (vx[j] + vx[j + 1])
        end
        vx_left_bound[j] = vx_left
        vx_right_bound[j] = vx_right
        dvx[j] = vx_right - vx_left
    end

    vr_over_dvr = Vector{Float64}(undef, nvr)
    @inbounds for i in 1:nvr
        vr_over_dvr[i] = vr[i] / dvr[i]
    end

    volume = Matrix{Float64}(undef, nvr, nvx)
    vmag_squared = Matrix{Float64}(undef, nvr, nvx)
    @inbounds for j in 1:nvx, i in 1:nvr
        volume[i, j] = dvr_vol[i] * dvx[j]
        vmag_squared[i, j] = vr[i]^2 + vx[j]^2
    end

    vth_dvx = zeros(Float64, nvr + 2, nvx + 2)
    vx_dvx  = zeros(Float64, nvr + 2, nvx + 2)
    vr_dvr  = zeros(Float64, nvr + 2, nvx + 2)

    @inbounds for j in 1:nvx
        inv_dvx = 1.0 / dvx[j]
        vx_over_dvx = vx[j] * inv_dvx
        for i in 1:nvr
            vth_dvx[i + 1, j + 1] = inv_dvx
            vx_dvx[i + 1, j + 1] = vx_over_dvx
            vr_dvr[i + 1, j + 1] = vr_over_dvr[i]
        end
    end

    # --- Get positive and negative indices from vx ---
    vx_pos_start = 0
    vx_pos_end = 0
    vx_neg_start = 0
    vx_neg_end = 0
    @inbounds for j in 1:nvx
        if vx[j] > 0.0
            vx_pos_start == 0 && (vx_pos_start = j)
            vx_pos_end = j
        elseif vx[j] < 0.0
            vx_neg_start == 0 && (vx_neg_start = j)
            vx_neg_end = j
        end
    end

    vx_pos_start != 0 || throw(ArgumentError("vx must contain at least one positive value"))
    vx_neg_start != 0 || throw(ArgumentError("vx must contain at least one negative value"))

    return VSpaceDifferentials(
        dvr_vol,
        dvr_vol_h_order,
        dvx,
        dvr,
        vr_left_bound,
        vr_right_bound,
        vx_left_bound,
        vx_right_bound,
        volume,
        vth_dvx,
        vx_dvx,
        vr_dvr,
        vmag_squared,
        vx_pos_start,
        vx_pos_end,
        vx_neg_start,
        vx_neg_end,
    )
end

function Base.show(io::IO, vsd::VSpaceDifferentials)
    println(io, "Velocity Space Differentials:")
    println(io, "    dvr_vol: ", vsd.dvr_vol)
    println(io, "    dvr_vol_h_order: ", vsd.dvr_vol_h_order)
    println(io, "    dvx: ", vsd.dvx)
    println(io, "    dvr: ", vsd.dvr)
    println(io, "    vr_right_bound: ", vsd.vr_right_bound)
    println(io, "    vr_left_bound: ", vsd.vr_left_bound)
    println(io, "    vx_right_bound: ", vsd.vx_right_bound)
    println(io, "    vx_left_bound: ", vsd.vx_left_bound)
    println(io, "    volume: ", vsd.volume)
    println(io, "    vth_dvx: ", vsd.vth_dvx)
    println(io, "    vx_dvx: ", vsd.vx_dvx)
    println(io, "    vr_dvr: ", vsd.vr_dvr)
    println(io, "    vmag_squared: ", vsd.vmag_squared)
    println(io, "    positive vx index range: ", vsd.vx_pos_start, ", ", vsd.vx_pos_end)
    print(io, "    negative vx index range: ", vsd.vx_neg_start, ", ", vsd.vx_neg_end)
end
