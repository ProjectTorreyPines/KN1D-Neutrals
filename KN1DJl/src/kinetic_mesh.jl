struct KineticMesh
    mesh_type::String
    x::Vector{Float64}
    Ti::Vector{Float64}
    Te::Vector{Float64}
    ne::Vector{Float64}
    PipeDia::Vector{Float64}
    vx::Vector{Float64}
    vr::Vector{Float64}
    Tnorm::Float64
end

function KineticMesh(
    mesh_type::String,
    mu::Int,
    x::Vector{Float64},
    Ti::Vector{Float64},
    Te::Vector{Float64},
    n::Vector{Float64},
    PipeDia::Vector{Float64};
    jh=nothing,
    E0::Vector{Float64}=[0.0],
    fctr=nothing,
    config_path::String="./config.json",
)::KineticMesh
    nx = length(x)
    println("Generating Kinetic $(mesh_type) Mesh...")

    cfg = get_config(config_path)
    if mesh_type == "h"
        nv = cfg.kinetic_h.mesh_size
        fctr === nothing && (fctr = cfg.kinetic_h.grid_fctr)
    elseif mesh_type == "h2"
        nv = cfg.kinetic_h2.mesh_size
        fctr === nothing && (fctr = cfg.kinetic_h2.grid_fctr)
    else
        throw(ArgumentError("ERROR: Mesh type invalid: $mesh_type"))
    end
    fctrf = Float64(fctr)

    react_rate = Vector{Float64}(undef, nx)
    if mesh_type == "h"
        @inbounds for i in eachindex(n, Te, react_rate)
            react_rate[i] = n[i] * sigmav_ion_h0(Te[i])
        end
        v0 = sqrt(20.0 * Q / (mu * H_MASS))
    else
        @inbounds for i in eachindex(n, Te, react_rate)
            react_rate[i] = n[i] * (
                sigmav_ion_hh(Te[i]) +
                sigmav_h1s_h1s_hh(Te[i]) +
                sigmav_h1s_h2s_hh(Te[i])
            )
        end
        v0 = sqrt(8.0 * TWALL * Q / (π * 2.0 * mu * H_MASS))
    end

    y = zeros(Float64, nx)
    @inbounds for i in 2:nx
        y[i] = y[i - 1] - ((x[i] - x[i - 1]) * 0.5 * (react_rate[i] + react_rate[i - 1])) / v0
    end

    xmax_clip = maximum(x)
    if mesh_type == "h"
        expdown = max(-5.0, minimum(y))
        xmax = min(interp_1d(y, x, expdown; fill_value="extrapolate"), xmax_clip)
    else
        xmax = min(interp_1d(y, x, -10.0), xmax_clip)
    end
    xmin = x[1]

    xfine = Vector{Float64}(undef, 1001)
    @inbounds for i in eachindex(xfine)
        xfine[i] = clamp(xmin + (xmax - xmin) * (i - 1) / 1000, xmin, xmax_clip)
    end

    Ti_itp = linear_interp_1d(x, Ti; fill_value="extrapolate")
    Te_itp = linear_interp_1d(x, Te)
    n_itp = linear_interp_1d(x, n)
    PipeDia_itp = linear_interp_1d(x, PipeDia)

    Tifine = Vector{Float64}(undef, 1001)
    Tefine = Vector{Float64}(undef, 1001)
    nfine = Vector{Float64}(undef, 1001)
    PipeDiafine = Vector{Float64}(undef, 1001)
    evaluate!(Tifine, Ti_itp, xfine)
    evaluate!(Tefine, Te_itp, xfine)
    evaluate!(nfine, n_itp, xfine)
    evaluate!(PipeDiafine, PipeDia_itp, xfine)

    vx, vr, Tnorm = create_vr_vx_mesh(nv, Tifine; E0=E0)
    vth = sqrt(2.0 * Q * Tnorm / (mu * H_MASS))

    nxfine = length(xfine)
    gamma_wall = zeros(Float64, nxfine)
    vrmax = maximum(vr)
    @inbounds for i in eachindex(gamma_wall, PipeDiafine)
        if PipeDiafine[i] > 0.0
            gamma_wall[i] = 2.0 * vrmax * vth / PipeDiafine[i]
        end
    end

    react_rate_fine = Vector{Float64}(undef, nxfine)
    if mesh_type == "h"
        minVr = vth * minimum(vr)
        minE0 = 0.5 * H_MASS * minVr^2 / Q
        ion_rate_option = cfg.kinetic_h.ion_rate
        ioniz_rate = Vector{Float64}(undef, nxfine)

        if ion_rate_option == "collrad"
            @inbounds for i in eachindex(nfine, Tefine, ioniz_rate)
                ioniz_rate[i] = collrad_sigmav_ion_h0(nfine[i], Tefine[i])
            end
        elseif ion_rate_option == "jh"
            jh === nothing && (jh = JohnsonHinnov())
            @inbounds for i in eachindex(nfine, Tefine, ioniz_rate)
                ioniz_rate[i] = jhs_coef(jh, nfine[i], Tefine[i]; no_null=true)
            end
        else
            @inbounds for i in eachindex(Tefine, ioniz_rate)
                ioniz_rate[i] = sigmav_ion_h0(Tefine[i])
            end
        end

        @inbounds for i in eachindex(nfine, Tifine, react_rate_fine, gamma_wall, ioniz_rate)
            react_rate_fine[i] =
                nfine[i] * (ioniz_rate[i] + sigmav_cx_h0(Tifine[i], minE0)) + gamma_wall[i]
        end
    else
        @inbounds for i in eachindex(nfine, Tefine, Tifine, react_rate_fine, gamma_wall)
            react_rate_fine[i] =
                nfine[i] * (
                    sigmav_ion_hh(Tefine[i]) +
                    sigmav_h1s_h1s_hh(Tefine[i]) +
                    sigmav_h1s_h2s_hh(Tefine[i]) +
                    0.1 * sigmav_cx_hh(Tifine[i], Tifine[i])
                ) + gamma_wall[i]
        end
    end

    dx_max = Vector{Float64}(undef, nxfine)
    vrmin = minimum(vr)
    dx_cap = 0.02 * fctrf
    @inbounds for i in eachindex(dx_max, react_rate_fine)
        dx_max[i] = min(fctrf * 0.8 * (2.0 * vth * vrmin / react_rate_fine[i]), dx_cap)
    end

    dx_itp = linear_interp_1d(xfine, dx_max; fill_value="extrapolate", assume_sorted=true)
    x_rev = Float64[xmax]
    sizehint!(x_rev, 256)
    xpt = xmax
    while xpt > xmin
        push!(x_rev, xpt)
        dxpt1 = dx_itp(xpt)
        dxpt2 = dxpt1
        xpt_test = xpt - dxpt1
        if xpt_test > xmin
            dxpt2 = dx_itp(xpt_test)
        end
        xpt -= min(dxpt1, dxpt2)
    end
    reverse!(x_rev)
    xH = vcat(xmin, x_rev[1:end-1])

    Ti_fine_itp = linear_interp_1d(xfine, Tifine; assume_sorted=true)
    Te_fine_itp = linear_interp_1d(xfine, Tefine; assume_sorted=true)
    n_fine_itp = linear_interp_1d(xfine, nfine; assume_sorted=true)
    PipeDia_fine_itp = linear_interp_1d(xfine, PipeDiafine; assume_sorted=true)

    nh = length(xH)
    TiH = Vector{Float64}(undef, nh)
    TeH = Vector{Float64}(undef, nh)
    neH = Vector{Float64}(undef, nh)
    PipeDiaH = Vector{Float64}(undef, nh)
    evaluate!(TiH, Ti_fine_itp, xH)
    evaluate!(TeH, Te_fine_itp, xH)
    evaluate!(neH, n_fine_itp, xH)
    evaluate!(PipeDiaH, PipeDia_fine_itp, xH)

    vx, vr, Tnorm = create_vr_vx_mesh(nv, TiH; E0=E0)

    return KineticMesh(mesh_type, xH, TiH, TeH, neH, PipeDiaH, vx, vr, Tnorm)
end

function create_vr_vx_mesh(
    nv::Int,
    Ti::Vector{Float64};
    E0::Vector{Float64}=[0.0],
    Tmax::Float64=0.0,
)::Tuple{Vector{Float64},Vector{Float64},Float64}
    n_extra = count(>(0.0), E0)
    Ti_work = Vector{Float64}(undef, length(Ti) + n_extra)
    k = 0
    @inbounds for t in Ti
        k += 1
        Ti_work[k] = t
    end
    @inbounds for e in E0
        if e > 0.0
            k += 1
            Ti_work[k] = e
        end
    end
    resize!(Ti_work, k)

    if Tmax > 0.0
        kept = 0
        @inbounds for i in eachindex(Ti_work)
            if Ti_work[i] < Tmax
                kept += 1
                Ti_work[kept] = Ti_work[i]
            end
        end
        resize!(Ti_work, kept)
    end

    maxTi = maximum(Ti_work)
    minTi = minimum(Ti_work)
    total = 0.0
    count_valid = 0
    @inbounds for t in Ti_work
        if !isnan(t)
            total += t
            count_valid += 1
        end
    end
    count_valid > 0 || throw(ArgumentError("Ti contains no valid temperatures"))
    Tnorm = total / count_valid

    v = Vector{Float64}(undef, nv + 1)
    vmax = 3.5
    if (maxTi - minTi) <= (0.1 * maxTi)
        @inbounds for i in 1:nv+1
            v[i] = ((i - 1) * vmax) / nv
        end
    else
        r = sqrt(minTi / maxTi)
        g = 2.0 * nv * r / (1.0 - r)
        b = vmax / (nv * (nv + g))
        @inbounds for i in 1:nv+1
            m = i - 1
            v[i] = (g * b) * m + b * m^2
        end
    end

    @inbounds for e in E0
        if e > 0.0
            v0 = sqrt(e / Tnorm)
            idx = searchsortedfirst(v, v0)
            if idx <= length(v)
                insert!(v, idx, v0)
            else
                push!(v, v0)
            end
        end
    end

    vr = v[2:end]
    nvr = length(vr)
    vx = Vector{Float64}(undef, 2 * nvr)
    @inbounds for i in 1:nvr
        vx[i] = -vr[nvr - i + 1]
        vx[nvr + i] = vr[i]
    end
    return vx, vr, Tnorm
end

function return_mesh_info(mesh::KineticMesh)::String
    string = "Kinetic Mesh: \n"
    string *= "    x: " * string(mesh.x) * "\n"
    string *= "    Ti: " * string(mesh.Ti) * "\n"
    string *= "    Te: " * string(mesh.Te) * "\n"
    string *= "    ne: " * string(mesh.ne) * "\n"
    string *= "    PipeDia: " * string(mesh.PipeDia) * "\n"
    string *= "    vx: " * string(mesh.vx) * "\n"
    string *= "    vr: " * string(mesh.vr) * "\n"
    string *= "    Tnorm: " * string(mesh.Tnorm) * "\n"
    return string
end
