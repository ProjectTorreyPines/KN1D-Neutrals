struct KH2Collisions
    # Enabled collision channels for the molecular-H2 solve.
    H2_H_EL::Bool
    H2_H2_EL::Bool
    H2_P_EL::Bool
    H2_P_CX::Bool
    SIMPLE_CX::Bool
end

function KH2Collisions(config::CollisionConfig)::KH2Collisions
    return KH2Collisions(
        config.H2_H_EL,
        config.H2_H2_EL,
        config.H2_P_EL,
        config.H2_P_CX,
        config.SIMPLE_CX,
    )
end

struct KH2MeshEqCoefficients
    # Directional transport recurrence coefficients for the discretized steady
    # molecular kinetic equation. A/B/F are used for vx > 0 sweeps; C/D/G for
    # vx < 0 sweeps.
    A::KHArray3
    B::KHArray3
    C::KHArray3
    D::KHArray3
    F::KHArray3
    G::KHArray3
end

struct KH2CollisionType{T}
    # Collision quantities grouped by molecular partner.
    H2_H2::T
    H2_P::T
    H2_H::T
end

struct KH2Results
    # Public result returned by the eventual molecular run_procedure.
    fH2::KHArray3
    nHP::Vector{Float64}
    THP::Vector{Float64}
    nH2::Vector{Float64}
    GammaxH2::Vector{Float64}
    VxH2::Vector{Float64}
    pH2::Vector{Float64}
    TH2::Vector{Float64}
    qxH2::Vector{Float64}
    qxH2_total::Vector{Float64}
    Sloss::Vector{Float64}
    QH2::Vector{Float64}
    RxH2::Vector{Float64}
    QH2_total::Vector{Float64}
    AlbedoH2::Float64
    WallH2::Vector{Float64}
    fSH::KHArray3
    SH::Vector{Float64}
    SP::Vector{Float64}
    SHP::Vector{Float64}
    NuE::Vector{Float64}
    NuDis::Vector{Float64}
    ESH::Matrix{Float64}
    Eaxis::Vector{Float64}
end

mutable struct KineticH2Output
    # Additional diagnostic channels mirroring the Python Kinetic_H2_Output
    # common block.
    piH2_xx::Vector{Float64}
    piH2_yy::Vector{Float64}
    piH2_zz::Vector{Float64}
    RxH2CX::Vector{Float64}
    RxH_H2::Vector{Float64}
    RxP_H2::Vector{Float64}
    RxW_H2::Vector{Float64}
    EH2CX::Vector{Float64}
    EH_H2::Vector{Float64}
    EP_H2::Vector{Float64}
    EW_H2::Vector{Float64}
    Epara_PerpH2_H2::Vector{Float64}
end

function KineticH2Output(nx::Integer)::KineticH2Output
    n = Int(nx)
    return KineticH2Output((zeros(Float64, n) for _ in 1:12)...)
end

mutable struct KineticH2Errors
    Max_dx::Union{Nothing,Vector{Float64}}
    vbar_error::Union{Nothing,Float64}
    mesh_error::Union{Nothing,Vector{Float64}}
    moment_error::Union{Nothing,Vector{Float64}}
    C_Error::Union{Nothing,Vector{Float64}}
    CX_Error::Union{Nothing,Vector{Float64}}
    Swall_Error::Union{Nothing,Vector{Float64}}
    H2_H2_error::Union{Nothing,Vector{Float64}}
    Source_Error::Union{Nothing,Vector{Float64}}
    qxH2_total_error::Union{Nothing,Vector{Float64}}
    QH2_total_error::Union{Nothing,Vector{Float64}}
end

KineticH2Errors() = KineticH2Errors(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)

mutable struct KineticH2Input
    vx_s::Union{Nothing,Vector{Float64}}
    vr_s::Union{Nothing,Vector{Float64}}
    x_s::Union{Nothing,Vector{Float64}}
    Tnorm_s::Union{Nothing,Float64}
    mu_s::Union{Nothing,Int}
    Ti_s::Union{Nothing,Vector{Float64}}
    Te_s::Union{Nothing,Vector{Float64}}
    n_s::Union{Nothing,Vector{Float64}}
    vxi_s::Union{Nothing,Vector{Float64}}
    fH2BC_s::Union{Nothing,Matrix{Float64}}
    GammaxH2BC_s::Union{Nothing,Float64}
    NuLoss_s::Union{Nothing,Vector{Float64}}
    PipeDia_s::Union{Nothing,Vector{Float64}}
    fH_s::Union{Nothing,KHArray3}
    SH2_s::Union{Nothing,Vector{Float64}}
    fH2_s::Union{Nothing,KHArray3}
    nHP_s::Union{Nothing,Vector{Float64}}
    THP_s::Union{Nothing,Vector{Float64}}
    SIMPLE_CX_s::Union{Nothing,Bool}
    Sawada_s::Union{Nothing,Bool}
    H2_H2_EL_s::Union{Nothing,Bool}
    H2_P_EL_s::Union{Nothing,Bool}
    H2_H_EL_s::Union{Nothing,Bool}
    H2_P_CX_s::Union{Nothing,Bool}
    ni_correct_s::Union{Nothing,Bool}
end

function KineticH2Input()::KineticH2Input
    return KineticH2Input((nothing for _ in 1:25)...)
end

mutable struct KineticH2Internal
    # Static and dynamic cached arrays for the molecular solve. Many fields are
    # populated by later translation stages; keeping them concrete/narrow here
    # makes those stages easier to port without Python-style dictionaries.
    vr2vx2::Union{Nothing,KHArray3}
    vr2vx_vxi2::Union{Nothing,KHArray3}
    fw_hat::Union{Nothing,Matrix{Float64}}
    fi_hat::Union{Nothing,KHArray3}
    fHp_hat::Union{Nothing,KHArray3}
    EH2_P::Union{Nothing,KHArray3}
    sigv::Union{Nothing,Matrix{Float64}}
    Alpha_Loss::Union{Nothing,Vector{Float64}}
    v_v2::Union{Nothing,KHArray5}
    v_v::Union{Nothing,KHArray5}
    vr2_vx2::Union{Nothing,KHArray5}
    vx_vx::Union{Nothing,Matrix{Float64}}
    Vr2pidVrdVx::Union{Nothing,Matrix{Float64}}
    SIG_CX::Union{Nothing,Matrix{Float64}}
    SIG_H2_H2::Union{Nothing,Matrix{Float64}}
    SIG_H2_H::Union{Nothing,Matrix{Float64}}
    SIG_H2_P::Union{Nothing,Matrix{Float64}}
    Alpha_CX::Union{Nothing,KHArray3}
    Alpha_H2_H::Union{Nothing,KHArray3}
    Alpha_H2_P::Union{Nothing,KHArray3}
    MH2_H2_sum::Union{Nothing,KHArray3}
    Delta_nH2s::Float64
end

function KineticH2Internal()::KineticH2Internal
    return KineticH2Internal((nothing for _ in 1:21)..., 0.0)
end

mutable struct KineticH2HMoments
    # Moments of the supplied atomic-H distribution, used by H2-H elastic
    # collision coupling.
    nH::Union{Nothing,Vector{Float64}}
    VxH::Union{Nothing,Vector{Float64}}
    TH::Union{Nothing,Vector{Float64}}
end

KineticH2HMoments() = KineticH2HMoments(nothing, nothing, nothing)

mutable struct KineticH2
    # Main solver state for molecular hydrogen transport.
    config::KN1DConfig
    collisions::KH2Collisions
    DeltaVx_tol::Float64
    Wpp_tol::Float64
    CI_Test::Bool
    Do_Alpha_CX_Test::Bool
    sawada::Bool
    compute_h_source::Bool
    ni_correct::Bool
    truncate::Float64
    max_gen::Int
    compute_errors::Bool
    debrief_level::Int
    debug::Int
    mesh::KineticMesh
    mu::Int
    vxi::Vector{Float64}
    fH2BC::Matrix{Float64}
    GammaxH2BC::Float64
    NuLoss::Vector{Float64}
    nvr::Int
    nvx::Int
    nx::Int
    vx_neg::Vector{Int}
    vx_pos::Vector{Int}
    vx_zero::Vector{Int}
    vth::Float64
    vr2_2vx2_2D::Matrix{Float64}
    dvr_vol::Vector{Float64}
    dvr_vol_h_order::Vector{Float64}
    dvx::Vector{Float64}
    fH2BC_input::Matrix{Float64}
    Eaxis::Vector{Float64}
    dEaxis::Vector{Float64}
    input::KineticH2Input
    internal::KineticH2Internal
    output::KineticH2Output
    h_moments::KineticH2HMoments
    errors::KineticH2Errors
end

function KineticH2(
    mesh::KineticMesh,
    mu::Integer,
    vxi::AbstractVector{<:Real},
    fH2BC::AbstractMatrix{<:Real},
    GammaxH2BC::Real,
    NuLoss::AbstractVector{<:Real},
    SH2_initial::AbstractVector{<:Real};
    sawada::Bool=true,
    compute_h_source::Bool=false,
    ni_correct::Bool=false,
    truncate::Real=1.0e-4,
    max_gen::Integer=100,
    compute_errors::Bool=false,
    debrief::Integer=0,
    debug::Integer=0,
    config_path::AbstractString="./config.json",
    initialize_static::Bool=false,
)::KineticH2
    mesh.mesh_type == "h2" || throw(ArgumentError("KineticH2 requires a mesh with mesh_type == \"h2\""))

    nvr = length(mesh.vr)
    nvx = length(mesh.vx)
    nx = length(mesh.x)
    length(vxi) == nx || throw(DimensionMismatch("vxi length must match mesh.x length"))
    length(NuLoss) == nx || throw(DimensionMismatch("NuLoss length must match mesh.x length"))
    length(SH2_initial) == nx || throw(DimensionMismatch("SH2_initial length must match mesh.x length"))
    size(fH2BC) == (nvr, nvx) || throw(DimensionMismatch("fH2BC must have size (length(mesh.vr), length(mesh.vx))"))

    config = get_config(String(config_path))
    collisions = KH2Collisions(config.collisions)
    debrief_level = Int(debrief)
    debug_level = Int(debug)
    if debug_level > 0
        debrief_level = max(debrief_level, 1)
    end

    mu_i = Int(mu)
    vxi_f = Float64.(vxi)
    fH2BC_f = Float64.(fH2BC)
    NuLoss_f = Float64.(NuLoss)

    vx_neg = findall(<(0.0), mesh.vx)
    vx_pos = findall(>(0.0), mesh.vx)
    vx_zero = findall(==(0.0), mesh.vx)
    isempty(vx_pos) && throw(ArgumentError("mesh.vx must contain at least one positive velocity"))
    isempty(vx_neg) && throw(ArgumentError("mesh.vx must contain at least one negative velocity"))

    vth = sqrt(2.0 * Q * mesh.Tnorm / (mu_i * H_MASS))

    vr2_2vx2_2D = Matrix{Float64}(undef, nvr, nvx)
    @inbounds for j in 1:nvx, i in 1:nvr
        vr2_2vx2_2D[i, j] = mesh.vr[i]^2 - 2.0 * mesh.vx[j]^2
    end

    differentials = VSpaceDifferentials(mesh.vr, mesh.vx)
    Eaxis, dEaxis = _h2_energy_axis(mesh.vr, vth, mu_i)

    kh2 = KineticH2(
        config,
        collisions,
        0.01,
        0.001,
        config.kinetic_h2.ci_test,
        config.kinetic_h2.alpha_cx_test,
        sawada,
        compute_h_source,
        ni_correct,
        Float64(truncate),
        Int(max_gen),
        compute_errors && debrief_level > 0,
        debrief_level,
        debug_level,
        mesh,
        mu_i,
        vxi_f,
        fH2BC_f,
        Float64(GammaxH2BC),
        NuLoss_f,
        nvr,
        nvx,
        nx,
        vx_neg,
        vx_pos,
        vx_zero,
        vth,
        vr2_2vx2_2D,
        differentials.dvr_vol,
        differentials.dvr_vol_h_order,
        differentials.dvx,
        zeros(Float64, nvr, nvx),
        Eaxis,
        dEaxis,
        KineticH2Input(),
        KineticH2Internal(),
        KineticH2Output(nx),
        KineticH2HMoments(),
        KineticH2Errors(),
    )

    _init_fh2bc_input!(kh2)
    _test_init_parameters(kh2)
    initialize_static && _init_static_internals!(kh2, Float64.(SH2_initial))

    return kh2
end

function _h2_energy_axis(vr::Vector{Float64}, vth::Float64, mu::Int)::Tuple{Vector{Float64},Vector{Float64}}
    nvr = length(vr)
    Eaxis = Vector{Float64}(undef, nvr)
    scale = vth^2 * 0.5 * mu * H_MASS / Q
    @inbounds for i in 1:nvr
        Eaxis[i] = scale * vr[i]^2
    end

    e_extended = Vector{Float64}(undef, nvr + 1)
    copyto!(e_extended, 1, Eaxis, 1, nvr)
    e_extended[end] = 2.0 * Eaxis[end] - Eaxis[end - 1]

    midpoint = Vector{Float64}(undef, nvr + 1)
    midpoint[1] = 0.0
    @inbounds for i in 2:nvr + 1
        midpoint[i] = 0.5 * (e_extended[i - 1] + e_extended[i])
    end

    dEaxis = Vector{Float64}(undef, nvr)
    @inbounds for i in 1:nvr
        dEaxis[i] = midpoint[i + 1] - midpoint[i]
    end
    return Eaxis, dEaxis
end

function _init_fh2bc_input!(kh2::KineticH2)::Nothing
    fill!(kh2.fH2BC_input, 0.0)
    @inbounds for j in kh2.vx_pos, i in 1:kh2.nvr
        kh2.fH2BC_input[i, j] = kh2.fH2BC[i, j]
    end

    gamma_input = 1.0
    if abs(kh2.GammaxH2BC) > 0.0
        total = 0.0
        @inbounds for j in 1:kh2.nvx
            vx_dvx = kh2.mesh.vx[j] * kh2.dvx[j]
            for i in 1:kh2.nvr
                total += kh2.dvr_vol[i] * kh2.fH2BC_input[i, j] * vx_dvx
            end
        end
        gamma_input = kh2.vth * total
    end

    ratio = abs(kh2.GammaxH2BC) / gamma_input
    @inbounds for j in 1:kh2.nvx, i in 1:kh2.nvr
        kh2.fH2BC_input[i, j] *= ratio
    end
    if abs(ratio - 1.0) > 0.01 * kh2.truncate
        copyto!(kh2.fH2BC, kh2.fH2BC_input)
    end
    return nothing
end

function _test_init_parameters(kh2::KineticH2)::Nothing
    kh2.mu in (1, 2) || throw(ArgumentError("mu must be 1 for hydrogen or 2 for deuterium"))
    kh2.truncate > 0.0 || throw(ArgumentError("truncate must be positive"))
    kh2.max_gen > 0 || throw(ArgumentError("max_gen must be positive"))
    isempty(kh2.vx_zero) || throw(ArgumentError("mesh.vx must not contain exactly zero velocity points"))
    all(>=(0.0), kh2.NuLoss) || throw(ArgumentError("NuLoss must be non-negative"))
    return nothing
end

function _init_static_internals!(kh2::KineticH2, SH2_initial::Vector{Float64})::Nothing
    _init_grid!(kh2, SH2_initial)
    _init_protons!(kh2)
    _init_sigv!(kh2)
    _init_v_v2!(kh2)
    _init_sig_kernels!(kh2)
    return nothing
end

function _debrief_msg(kh2::KineticH2, message::AbstractString, threshold::Integer)::Nothing
    debrief("Kinetic_H2 => " * message, kh2.debrief_level > threshold)
    return nothing
end

function _init_grid!(kh2::KineticH2, SH2_initial::Vector{Float64})::Nothing
    vr = kh2.mesh.vr
    vx = kh2.mesh.vx
    nx = kh2.nx

    vr2vx2 = Array{Float64,3}(undef, kh2.nvr, kh2.nvx, nx)
    vr2vx_vxi2 = similar(vr2vx2)
    EH2_P = similar(vr2vx2)

    energy_scale = H_MASS * kh2.vth^2 / Q
    @inbounds for k in 1:nx
        shift = kh2.vxi[k] / kh2.vth
        for j in 1:kh2.nvx
            vxj = vx[j]
            shifted2 = (vxj - shift)^2
            vx2 = vxj^2
            for i in 1:kh2.nvr
                vr2 = vr[i]^2
                vr2vx2[i, j, k] = vr2 + vx2
                rel = vr2 + shifted2
                vr2vx_vxi2[i, j, k] = rel
                EH2_P[i, j, k] = clamp(energy_scale * rel, 0.1, 2.0e4)
            end
        end
    end

    kh2.internal.vr2vx2 = vr2vx2
    kh2.internal.vr2vx_vxi2 = vr2vx_vxi2
    kh2.internal.EH2_P = EH2_P

    if sum(SH2_initial) > 0.0 || sum(kh2.mesh.PipeDia) > 0.0
        maxwell = create_shifted_maxwellian(kh2.mesh.vr, kh2.mesh.vx, [TWALL], [0.0], kh2.mu, 2, kh2.mesh.Tnorm)
        kh2.internal.fw_hat = copy(@view maxwell[:, :, 1])
    end
    return nothing
end

function _init_protons!(kh2::KineticH2)::Nothing
    kh2.internal.fi_hat = create_shifted_maxwellian(kh2.mesh.vr, kh2.mesh.vx, kh2.mesh.Ti, kh2.vxi, kh2.mu, 1, kh2.mesh.Tnorm)
    return nothing
end

function _init_sigv!(kh2::KineticH2)::Nothing

    nx = kh2.nx
    _debrief_msg(kh2, "Computing sigv", 1)
    sigv = zeros(Float64, nx, 11)

    # Python keeps column 0 unused and stores reaction Rn in sigv[:, n].
    # Julia preserves that convention as column n+1.
    @inbounds for k in 1:nx
        Te = kh2.mesh.Te[k]
        sigv[k, 2] = sigmav_ion_hh(Te)
        sigv[k, 3] = sigmav_h1s_h1s_hh(Te)
        sigv[k, 4] = sigmav_h1s_h2s_hh(Te)
        sigv[k, 5] = sigmav_p_h1s_hh(Te)
        sigv[k, 6] = sigmav_h2p_h2s_hh(Te)
        sigv[k, 7] = sigmav_h1s_hn3_hh(Te)
        sigv[k, 8] = sigmav_p_h1s_hp(Te)
        sigv[k, 9] = sigmav_p_hn2_hp(Te)
        sigv[k, 10] = sigmav_p_p_hp(Te)
        sigv[k, 11] = sigmav_h1s_hn_hp(Te)
    end

    if kh2.sawada
        @inbounds for k in 1:nx
            sigv[k, 2] *= 3.7 / 2.0
        end

        # Construct table
        Te_table = log.([5.0, 20.0, 100.0])
        Ne_table = log.([1e14, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22])

        fctr_table = zeros(Float64, 7, 3)

        fctr_table[:, 1] .= [2.2, 2.2, 2.1, 1.9, 1.2, 1.1, 1.05] ./ 5.3
        fctr_table[:, 2] .= [5.1, 5.1, 4.3, 3.1, 1.5, 1.25, 1.25] ./ 10.05
        fctr_table[:, 3] .= [1.3, 1.3, 1.1, 0.8, 0.38, 0.24, 0.22] ./ 2.1

        _Te = clamp.(kh2.mesh.Te, 5.0, 100.0)
        _n  = clamp.(kh2.mesh.ne, 1e14, 1e22)

        fctr = path_interp_2d(
            fctr_table,
            Ne_table,
            Te_table,
            log.(_n),
            log.(_Te),
        )

        sigv[:, 2] .= (1.0 .+ fctr) .* sigv[:, 2]

        sigv[:, 4] .= sigv[:, 4] .* (1.0/0.6)
    end

    Alpha_Loss = Vector{Float64}(undef, kh2.nx)
    @inbounds for k in 1:kh2.nx
        Alpha_Loss[k] = kh2.mesh.ne[k] * sum(@view sigv[k, 2:7]) / kh2.vth
    end

    kh2.internal.sigv = sigv
    kh2.internal.Alpha_Loss = Alpha_Loss
    return nothing
end

function run_procedure(kh2::KineticH2; fH=nothing, SH2=nothing, fH2=nothing, nHP=nothing, THP=nothing)::KH2Results
    nvr, nvx, nx = kh2.nvr, kh2.nvx, kh2.nx

    fH_f = _as_array3_or_zeros(fH, nvr, nvx, nx, "fH")
    fH2_f = _as_array3_or_zeros(fH2, nvr, nvx, nx, "fH2")
    SH2_f = _as_vector_or(SH2, nx, 0.0, "SH2")
    nHP_f = _as_vector_or(nHP, nx, 0.0, "nHP")
    THP_f = _as_vector_or(THP, nx, 3.0, "THP")

    # Static internals depend on wall/source availability, so build lazily here
    # using the actual SH2 passed to the run when the constructor did not do it.
    if kh2.internal.vr2vx2 === nothing ||
       kh2.internal.fi_hat === nothing ||
       kh2.internal.sigv === nothing ||
       kh2.internal.SIG_CX === nothing ||
       kh2.internal.fw_hat === nothing
        _init_static_internals!(kh2, SH2_f)
    end

    # Match Python: H2-H elastic collisions are disabled for a zero atomic-H distribution.
    kh2.collisions = KH2Collisions(
        sum(fH_f) > 0.0 && kh2.config.collisions.H2_H_EL,
        kh2.collisions.H2_H2_EL,
        kh2.collisions.H2_P_EL,
        kh2.collisions.H2_P_CX,
        kh2.collisions.SIMPLE_CX,
    )

    @inbounds for j in kh2.vx_pos, i in 1:nvr
        fH2_f[i, j, 1] = kh2.fH2BC_input[i, j]
    end

    Do_Alpha_CX, Do_Alpha_H2_P = _compute_dynamic_internals!(kh2, fH_f, fH2_f, nHP_f, THP_f)

    nH2 = Vector{Float64}(undef, nx)
    @inbounds for k in 1:nx
        s = 0.0
        for j in 1:nvx
            dvxj = kh2.dvx[j]
            for i in 1:nvr
                s += kh2.dvr_vol[i] * fH2_f[i, j, k] * dvxj
            end
        end
        nH2[k] = s
    end

    gamma_wall = zeros(Float64, nvr, nvx, nx)
    @inbounds for k in 1:nx
        if kh2.mesh.PipeDia[k] > 0.0
            inv_dia = 2.0 / kh2.mesh.PipeDia[k]
            for j in 1:nvx, i in 1:nvr
                gamma_wall[i, j, k] = kh2.mesh.vr[i] * inv_dia
            end
        end
    end

    fH2_out, alpha_c, Beta_CX_sum, Swall_sum, collision_freqs, m_sums =
        _run_iteration_scheme(kh2, fH2_f, nH2, nHP_f, THP_f, SH2_f, gamma_wall, Do_Alpha_CX, Do_Alpha_H2_P)

    results = _compile_results(kh2, fH2_out, SH2_f, gamma_wall, alpha_c, Beta_CX_sum, Swall_sum, collision_freqs, m_sums)

    kh2.input.fH_s = fH_f
    kh2.input.SH2_s = SH2_f
    kh2.input.fH2_s = results.fH2
    kh2.input.nHP_s = results.nHP
    kh2.input.THP_s = results.THP
    kh2.input.ni_correct_s = kh2.ni_correct

    _debrief_msg(kh2, "Finished", 0)
    return results
end

function _unported_kinetic_h2(name::Symbol)
    throw(ErrorException("KineticH2.$name has not been ported yet"))
end

function _init_v_v2!(kh2::KineticH2)::Nothing
    _debrief_msg(kh2, "Computing compact velocity-space kernel geometry", 1)

    vx = kh2.mesh.vx
    nvr = kh2.nvr
    nvx = kh2.nvx

    vx_diff = Matrix{Float64}(undef, nvx, nvx)

    @inbounds for j2 in 1:nvx
        vx2 = vx[j2]
        for j1 in 1:nvx
            vx_diff[j1, j2] = vx[j1] - vx2
        end
    end

    Vr2pidVrdVx = Matrix{Float64}(undef, nvr, nvx)

    @inbounds for j2 in 1:nvx
        dvx2 = kh2.dvx[j2]
        for i2 in 1:nvr
            Vr2pidVrdVx[i2, j2] = kh2.dvr_vol[i2] * dvx2
        end
    end

    kh2.internal.vx_vx = vx_diff
    kh2.internal.Vr2pidVrdVx = Vr2pidVrdVx

    kh2.internal.v_v2 = nothing
    kh2.internal.v_v = nothing
    kh2.internal.vr2_vx2 = nothing

    return nothing
end

function _init_static_internals!(kh2::KineticH2)::Nothing
    return _init_static_internals!(kh2, zeros(Float64, kh2.nx))
end

function _init_sig_kernels!(kh2::KineticH2)::Nothing
    _debrief_msg(kh2, "Computing fused SIG_CX, SIG_H2_H2, SIG_H2_H, and SIG_H2_P", 1)

    vr = kh2.mesh.vr
    vx = kh2.mesh.vx
    nvr = kh2.nvr
    nvx = kh2.nvx
    ntheta = KH_NTHETA
    nvel = nvr * nvx

    scale_cx = H_MASS * kh2.vth^2 / Q
    scale_h2_h2 = H_MASS * kh2.mu * kh2.vth^2 / Q
    scale_h2_h = 0.5 * H_MASS * kh2.vth^2 / Q

    SIG_CX = Array{Float64, 2}(undef, nvel, nvel)
    SIG_H2_H2 = Array{Float64, 2}(undef, nvel, nvel)
    SIG_H2_H = Array{Float64, 2}(undef, nvel, nvel)
    SIG_H2_P = Array{Float64, 2}(undef, nvel, nvel)

    Threads.@threads for j2 in 1:nvx
        @inbounds begin
            vx2 = vx[j2]
            for i2 in 1:nvr
                vr2 = vr[i2]
                vr2sq = vr2 * vr2
                col = i2 + (j2 - 1) * nvr
                weight = kh2.internal.Vr2pidVrdVx[i2, j2]

                for j1 in 1:nvx
                    dvx = vx[j1] - vx2
                    dvx2 = dvx * dvx
                    for i1 in 1:nvr
                        vr1 = vr[i1]
                        vr1sq = vr1 * vr1
                        row = i1 + (j1 - 1) * nvr

                        s_cx = 0.0
                        s_h2_h2 = 0.0
                        s_h2_h = 0.0
                        s_h2_p = 0.0

                        for c in 1:ntheta
                            base = vr1sq + vr2sq - 2.0 * vr1 * vr2 * KH_COS_THETA[c]
                            vv2 = base + dvx2
                            vv = sqrt(vv2)
                            dtheta = KH_DTHETA[c]

                            s_cx += vv * sigma_cx_hh(vv2 * scale_cx) * dtheta
                            s_h2_h2 += (base - 2.0 * dvx2) * vv * sigma_el_hh_hh(vv2 * scale_h2_h2, vis=true) * dtheta / 8.0
                            s_h2_h += vv * sigma_el_h_hh(vv2 * scale_h2_h) * dtheta
                            s_h2_p += vv * sigma_el_p_hh(vv2 * scale_h2_h) * dtheta
                        end

                        SIG_CX[row, col] = weight * s_cx
                        SIG_H2_H2[row, col] = weight * s_h2_h2
                        SIG_H2_H[row, col] = weight * dvx * s_h2_h
                        SIG_H2_P[row, col] = weight * dvx * s_h2_p
                    end
                end
            end
        end
    end

    kh2.internal.SIG_CX = SIG_CX
    kh2.internal.SIG_H2_H2 = SIG_H2_H2
    kh2.internal.SIG_H2_H = SIG_H2_H
    kh2.internal.SIG_H2_P = SIG_H2_P

    return nothing
end

function _compute_alpha_cx!(kh2::KineticH2, nHP::Vector{Float64}, THP::Vector{Float64})::Nothing
    _debrief_msg(kh2, "Computing Alpha_CX", 1)

    nvr = kh2.nvr
    nvx = kh2.nvx
    nx = kh2.nx

    length(nHP) == nx || throw(DimensionMismatch("nHP length must match kh2.nx"))
    length(THP) == nx || throw(DimensionMismatch("THP length must match kh2.nx"))

    Alpha_CX = Array{Float64, 3}(undef, nvr, nvx, nx)

    kh2.internal.fHp_hat = create_shifted_maxwellian(kh2.mesh.vr, kh2.mesh.vx, THP, kh2.vxi, kh2.mu, 2, kh2.mesh.Tnorm)

    if kh2.collisions.SIMPLE_CX
        @inbounds for k in 1:nx
            thp_mu = THP[k] / kh2.mu
            nHP_k = nHP[k]
            for j in 1:nvx
                for i in 1:nvr
                    Alpha_CX[i, j, k] =
                        sigmav_cx_hh(thp_mu, kh2.internal.EH2_P[i, j, k]) / kh2.vth * nHP_k
                end
            end
        end
    else
        kh2.internal.SIG_CX === nothing && throw(ArgumentError("SIG_CX must be initialized before direct Alpha_CX computation"))
        work = Vector{Float64}(undef, nvr * nvx)
        _mul_scaled_kernel_slices!(Alpha_CX, kh2.internal.SIG_CX, kh2.internal.fHp_hat, nHP, work)
    end

    kh2.internal.Alpha_CX = Alpha_CX
    return nothing
end

function _compute_alpha_h2_h!(kh2::KineticH2, fH::Array{Float64, 3})::Nothing
    _debrief_msg(kh2, "Computing Alpha_H2_H", 1)
    size(fH) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("fH must have size (kh2.nvr, kh2.nvx, kh2.nx)"))

    nvr = kh2.nvr
    nvx = kh2.nvx
    nx = kh2.nx

    Alpha_H2_H = Array{Float64, 3}(undef, nvr, nvx, nx)
    work = Vector{Float64}(undef, nvr * nvx)
    _mul_kernel_slices!(Alpha_H2_H, kh2.internal.SIG_H2_H, fH, work)

    kh2.internal.Alpha_H2_H = Alpha_H2_H
    return nothing
end

function _compute_alpha_h2_p!(kh2::KineticH2, nHP::Vector{Float64})::Nothing
    _debrief_msg(kh2, "Computing Alpha_H2_P", 1)

    ni = kh2.mesh.ne
    if kh2.ni_correct
        ni = similar(nHP)
        @inbounds for k in 1:kh2.nx
            ni[k] = max(kh2.mesh.ne[k] - nHP[k], 0.0)
        end
    end

    Alpha_H2_P = Array{Float64, 3}(undef, kh2.nvr, kh2.nvx, kh2.nx)
    work = Vector{Float64}(undef, kh2.nvr * kh2.nvx)
    _mul_scaled_kernel_slices!(Alpha_H2_P, kh2.internal.SIG_H2_P, kh2.internal.fi_hat, ni, work)
    kh2.internal.Alpha_H2_P = Alpha_H2_P
    return nothing
end

function _compute_fh_moments!(kh2::KineticH2, fH::Array{Float64,3})::Nothing
    _debrief_msg(kh2, "Computing vx and T moments of fH", 1)
    size(fH) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("fH must have size (kh2.nvr, kh2.nvx, kh2.nx)"))

    vr = kh2.mesh.vr
    vx = kh2.mesh.vx
    dvr_vol = kh2.dvr_vol
    dvx = kh2.dvx
    nvr = kh2.nvr
    nvx = kh2.nvx
    nx = kh2.nx
    vth = kh2.vth

    nH = zeros(Float64, nx)
    VxH = zeros(Float64, nx)
    TH = fill(1.0, nx)

    @inbounds for k in 1:nx
        n_sum = 0.0
        vx_sum = 0.0

        for j in 1:nvx
            vxj = vx[j]
            wvx = dvx[j]
            for i in 1:nvr
                w = dvr_vol[i] * wvx
                f = fH[i, j, k]
                n_sum += f * w
                vx_sum += f * vxj * w
            end
        end

        nH[k] = n_sum
        n_sum <= 0.0 && continue

        VxH_k = vth * vx_sum / n_sum
        VxH[k] = VxH_k

        vx_shift = VxH_k / vth
        temp_sum = 0.0

        for j in 1:nvx
            vxran = vx[j] - vx_shift
            vxran2 = vxran * vxran
            wvx = dvx[j]
            for i in 1:nvr
                vri = vr[i]
                vr2vx2_ran2 = vri * vri + vxran2
                temp_sum += vr2vx2_ran2 * fH[i, j, k] * dvr_vol[i] * wvx
            end
        end

        TH[k] = kh2.mu * H_MASS * vth^2 * temp_sum / (3.0 * Q * n_sum)
    end

    kh2.h_moments.nH = nH
    kh2.h_moments.VxH = VxH
    kh2.h_moments.TH = TH
    return nothing
end

function _compute_dynamic_internals!(kh2::KineticH2,
    fH::Array{Float64, 3}, 
    fH2::Array{Float64, 3},
    nHP::Vector{Float64}, 
    THP::Vector{Float64}
    )::Tuple{Bool,Bool}

    New_fH = true
    if (kh2.input.fH_s !== nothing) && kh2.input.fH_s == fH
        New_fH = false
    end

    New_H2_Seed = true
    if (kh2.input.fH2_s !== nothing) && kh2.input.fH2_s == fH2
        New_H2_Seed = false
    end

    New_HP_Seed = true
    if (kh2.input.nHP_s !== nothing) && (kh2.input.THP_s !== nothing) && kh2.input.nHP_s == nHP && kh2.input.THP_s == THP
        New_HP_Seed = false
    end

    New_ni_correct = true
    if kh2.input.ni_correct_s !== nothing && (kh2.input.ni_correct_s == kh2.ni_correct)
        New_ni_correct = false
    end

    Do_Alpha_CX = ((kh2.internal.Alpha_CX === nothing) || New_HP_Seed) && kh2.collisions.H2_P_CX
    Do_Alpha_H2_H = ((kh2.internal.Alpha_H2_H === nothing) || New_fH) && kh2.collisions.H2_H_EL
    Do_Alpha_H2_P = ((kh2.internal.Alpha_H2_P === nothing) || New_ni_correct) && kh2.collisions.H2_P_EL

    kh2.h_moments.nH = zeros(Float64, kh2.nx)
    kh2.h_moments.VxH = zeros(Float64, kh2.nx)
    kh2.h_moments.TH = fill(1.0, kh2.nx)

    #Checking sum fH is greater than zero to avoid unnecessary moment calculations and potential NaNs in the moments.
    if New_fH && (sum(fH) > 0.0)
        _compute_fh_moments!(kh2, fH)
    end
    if Do_Alpha_H2_H
        _compute_alpha_h2_h!(kh2, fH)
    end
    if New_H2_Seed
        kh2.internal.MH2_H2_sum = zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)
        kh2.internal.Delta_nH2s = 1.0
    end
    return Do_Alpha_CX, Do_Alpha_H2_P
end

function _compute_swall(kh2::KineticH2, fH2G::Array{Float64,3}, gamma_wall::Array{Float64,3})::Array{Float64,3}
    _debrief_msg(kh2, "Computing Swall", 1)
    size(fH2G) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("fH2G must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    size(gamma_wall) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("gamma_wall must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    kh2.internal.fw_hat === nothing && throw(ArgumentError("fw_hat must be initialized before computing Swall"))

    nvr = kh2.nvr
    nvx = kh2.nvx
    nx = kh2.nx

    Swall = zeros(Float64, nvr, nvx, nx)
    sum(gamma_wall) <= 0.0 && return Swall

    fw_hat = kh2.internal.fw_hat
    dvr_vol = kh2.dvr_vol
    dvx = kh2.dvx

    @inbounds for k in 1:nx
        wall_source = 0.0
        for j in 1:nvx
            dvxj = dvx[j]
            for i in 1:nvr
                wall_source += dvr_vol[i] * gamma_wall[i, j, k] * fH2G[i, j, k] * dvxj
            end
        end

        for j in 1:nvx
            for i in 1:nvr
                Swall[i, j, k] = fw_hat[i, j] * wall_source
            end
        end
    end

    return Swall
end

function _compute_beta_cx(kh2::KineticH2, fH2G::Array{Float64,3}, nHP::Vector{Float64})::Array{Float64,3}
    size(fH2G) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("fH2G must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    length(nHP) == kh2.nx || throw(DimensionMismatch("nHP length must match kh2.nx"))

    nvr = kh2.nvr
    nvx = kh2.nvx
    nx = kh2.nx
    nvel = nvr * nvx

    Beta_CX = zeros(Float64, nvr, nvx, nx)
    if kh2.collisions.H2_P_CX
        _debrief_msg(kh2, "Computing Beta CX", 1)
        kh2.internal.fHp_hat === nothing && throw(ArgumentError("fHp_hat must be initialized before computing Beta_CX"))
        if kh2.collisions.SIMPLE_CX
            kh2.internal.Alpha_CX === nothing && throw(ArgumentError("Alpha_CX must be initialized before simple Beta_CX computation"))
            @inbounds for k in 1:nx
                source = 0.0
                for j in 1:nvx
                    dvxj = kh2.dvx[j]
                    for i in 1:nvr
                        source += kh2.dvr_vol[i] * kh2.internal.Alpha_CX[i, j, k] * fH2G[i, j, k] * dvxj
                    end
                end
                for j in 1:nvx
                    for i in 1:nvr
                        Beta_CX[i, j, k] = kh2.internal.fHp_hat[i, j, k] * source
                    end
                end
            end
        else
            kh2.internal.SIG_CX === nothing && throw(ArgumentError("SIG_CX must be initialized before direct Beta_CX computation"))
            work = Vector{Float64}(undef, nvel)
            @inbounds for k in 1:nx
                fH2G_k = @view fH2G[:, :, k]
                for idx in 1:nvel
                    work[idx] = fH2G_k[idx]
                end

                beta_k = reshape(@view(Beta_CX[:, :, k]), nvel)
                mul!(beta_k, kh2.internal.SIG_CX, work)

                nHP_k = nHP[k]
                fHp_hat_k = @view kh2.internal.fHp_hat[:, :, k]
                for idx in 1:nvel
                    beta_k[idx] *= nHP_k * fHp_hat_k[idx]
                end
            end
        end
    end

    return Beta_CX
end

function _compute_mh_values(
    kh2::KineticH2,
    fH2G::Array{Float64,3},
    nH2::AbstractVector{Float64},
)::KH2CollisionType{Array{Float64,3}}
    size(fH2G) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("fH2G must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    length(nH2) == kh2.nx || throw(DimensionMismatch("nH2 length must match kh2.nx"))

    nvr = kh2.nvr
    nvx = kh2.nvx
    nx = kh2.nx
    vth = kh2.vth

    MH2_H2 = zeros(Float64, nvr, nvx, nx)
    MH2_P = zeros(Float64, nvr, nvx, nx)
    MH2_H = zeros(Float64, nvr, nvx, nx)
    VxH2G = zeros(Float64, nx)
    TH2G = zeros(Float64, nx)

    if kh2.collisions.H2_H2_EL || kh2.collisions.H2_P_EL || kh2.collisions.H2_H_EL
        @inbounds for k in 1:nx
            nH2k = nH2[k]
            vx_sum = 0.0

            for j in 1:nvx
                vxj = kh2.mesh.vx[j]
                dvxj = kh2.dvx[j]
                for i in 1:nvr
                    vx_sum += fH2G[i, j, k] * kh2.dvr_vol[i] * dvxj * vxj
                end
            end

            VxH2G[k] = vth * vx_sum / nH2k

            temp_sum = 0.0
            vx_shift = VxH2G[k] / vth
            for j in 1:nvx
                vxran2 = (kh2.mesh.vx[j] - vx_shift)^2
                dvxj = kh2.dvx[j]
                for i in 1:nvr
                    vr2vx2_ran2 = kh2.mesh.vr[i]^2 + vxran2
                    temp_sum += kh2.dvr_vol[i] * vr2vx2_ran2 * fH2G[i, j, k] * dvxj
                end
            end

            TH2G[k] = (2.0 * kh2.mu * H_MASS) * vth^2 * temp_sum / (3.0 * Q * nH2k)
        end

        if kh2.collisions.H2_H2_EL
            _debrief_msg(kh2, "Computing MH2_H2", 1)

            Maxwell = create_shifted_maxwellian(kh2.mesh.vr, kh2.mesh.vx, TH2G, VxH2G, kh2.mu, 2, kh2.mesh.Tnorm)
            @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
                MH2_H2[i, j, k] = Maxwell[i, j, k] * nH2[k]
            end
        end

        if kh2.collisions.H2_P_EL
            _debrief_msg(kh2, "Computing MH2_P", 1)

            vx_shift = Vector{Float64}(undef, nx)
            Tmaxwell = Vector{Float64}(undef, nx)
            @inbounds for k in 1:nx
                vx_shift[k] = (2.0 * VxH2G[k] + kh2.vxi[k]) / 3.0
                Tmaxwell[k] = TH2G[k] + (4.0 / 9.0) * (
                    kh2.mesh.Ti[k] - TH2G[k] +
                    kh2.mu * H_MASS * (kh2.vxi[k] - VxH2G[k])^2 / (6.0 * Q)
                )
            end
            Maxwell = create_shifted_maxwellian(kh2.mesh.vr, kh2.mesh.vx, Tmaxwell, vx_shift, kh2.mu, 2, kh2.mesh.Tnorm)
            @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
                MH2_P[i, j, k] = Maxwell[i, j, k] * nH2[k]
            end
        end

        if kh2.collisions.H2_H_EL
            _debrief_msg(kh2, "Computing MH2_H", 1)
            kh2.h_moments.VxH === nothing && throw(ArgumentError("H moments must be initialized before computing MH2_H"))
            kh2.h_moments.TH === nothing && throw(ArgumentError("H moments must be initialized before computing MH2_H"))

            vx_shift = Vector{Float64}(undef, nx)
            Tmaxwell = Vector{Float64}(undef, nx)
            @inbounds for k in 1:nx
                vxh = kh2.h_moments.VxH[k]
                th = kh2.h_moments.TH[k]
                vx_shift[k] = (2.0 * VxH2G[k] + vxh) / 3.0
                Tmaxwell[k] = TH2G[k] + (4.0 / 9.0) * (
                    th - TH2G[k] +
                    kh2.mu * H_MASS * (vxh - VxH2G[k])^2 / (6.0 * Q)
                )
            end
            Maxwell = create_shifted_maxwellian(kh2.mesh.vr, kh2.mesh.vx, Tmaxwell, vx_shift, kh2.mu, 2, kh2.mesh.Tnorm)
            @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
                MH2_H[i, j, k] = Maxwell[i, j, k] * nH2[k]
            end
        end
    end

    return KH2CollisionType(MH2_H2, MH2_P, MH2_H)
end

function _compute_omega_values(kh2::KineticH2, 
    fH2::Array{Float64,3}, 
    nH2::Vector{Float64}
    )::KH2CollisionType{Vector{Float64}}
    nvr, nvx, nx = kh2.nvr, kh2.nvx, kh2.nx

    Omega_H2_P = zeros(Float64, nx)
    Omega_H2_H2 = zeros(Float64, nx)
    Omega_H2_H = zeros(Float64, nx)
    vth = kh2.vth

    if any(nH2 .<= 0.0)
        return KH2CollisionType(Omega_H2_H2, Omega_H2_P, Omega_H2_H)
    end

    VxH2 = Vector{Float64}(undef, nx)

    if kh2.collisions.H2_P_EL || kh2.collisions.H2_H_EL || kh2.collisions.H2_H2_EL
        @inbounds for k in 1:nx
            vx_sum = 0.0
            for j in 1:nvx 
                vxj = kh2.mesh.vx[j]
                dvxj = kh2.dvx[j]
                for i in 1:nvr
                    f = fH2[i, j, k]
                    w = kh2.dvr_vol[i] * dvxj * vxj
                    vx_sum += f * w
                end
            end
            VxH2[k] = vth * vx_sum / nH2[k]
        end
    end 
    if kh2.collisions.H2_P_EL
        _debrief_msg(kh2, "Computing Omega_H2_P", 1)
        @inbounds for k in 1:nx
            # Avoid division by a near-zero drift speed while preserving sign,
            # matching the Python/IDL tolerance behavior.
            raw = (VxH2[k] - kh2.vxi[k]) / kh2.vth
            mag = max(abs(raw), kh2.DeltaVx_tol)
            delta = raw >= 0.0 ? mag : -mag

            s = 0.0
            for j in 1:nvx
                dvxj = kh2.dvx[j]
                for i in 1:nvr
                    s += kh2.dvr_vol[i] *
                        kh2.internal.Alpha_H2_P[i, j, k] *
                        fH2[i, j, k] *
                        dvxj
                end
            end

            Omega_H2_P[k] = max(s / (nH2[k] * delta), 0.0)
        end
    end
    if kh2.collisions.H2_H_EL
        _debrief_msg(kh2, "Computing Omega_H2_H", 1)
        @inbounds for k in 1:nx
            raw = (VxH2[k] - kh2.h_moments.VxH[k]) / kh2.vth
            mag = max(abs(raw), kh2.DeltaVx_tol)
            delta = raw >= 0.0 ? mag : -mag

            s = 0.0
            for j in 1:nvx
                dvxj = kh2.dvx[j]
                for i in 1:nvr
                    s += kh2.dvr_vol[i] *
                        kh2.internal.Alpha_H2_H[i, j, k] *
                        fH2[i, j, k] *
                        dvxj
                end
            end

            Omega_H2_H[k] = max(s / (nH2[k] * delta), 0.0)
        end
    end
    if kh2.collisions.H2_H2_EL
        _debrief_msg(kh2, "Computing Omega_H2_H2", 1)

        # H2-H2 elastic scattering uses a perpendicular/parallel energy exchange
        # moment. On later outer iterations, the accumulated MH2_H2 source from
        # the previous solve is used to form this balance.
        Wperp_paraH2 = zeros(Float64, nx)

        if kh2.internal.MH2_H2_sum === nothing || sum(kh2.internal.MH2_H2_sum) <= 0.0
            @inbounds for k in 1:nx
                s = 0.0
                VxH2_k = VxH2[k]

                for j in 1:nvx
                    vxran = kh2.mesh.vx[j] - VxH2_k
                    vxterm = 2.0 * vxran * vxran
                    dvxj = kh2.dvx[j]

                    for i in 1:nvr
                        vri = kh2.mesh.vr[i]
                        vr2_2vx_ran2 = vri * vri - vxterm

                        s += kh2.dvr_vol[i] *
                            vr2_2vx_ran2 *
                            fH2[i, j, k] *
                            dvxj
                    end
                end

                Wperp_paraH2[k] = s / nH2[k]
            end
        else
            @inbounds for k in 1:nx
                s = 0.0

                for j in 1:nvx
                    dvxj = kh2.dvx[j]

                    for i in 1:nvr
                        M_fH2 = kh2.internal.MH2_H2_sum[i, j, k] - fH2[i, j, k]

                        s += kh2.dvr_vol[i] *
                            kh2.vr2_2vx2_2D[i, j] *
                            M_fH2 *
                            dvxj
                    end
                end

                Wperp_paraH2[k] = -s / nH2[k]
            end
        end
        @inbounds for k in 1:nx
            s = 0.0

            for j in 1:nvx
                dvxj = kh2.dvx[j]

                for i in 1:nvr
                    row = i + (j - 1) * nvr

                    # Apply the flattened velocity-space collision kernel using
                    # Julia's column-major order, equivalent to NumPy
                    # reshape(..., order="F") in the reference implementation.
                    alpha_ij = 0.0
                    for j2 in 1:nvx
                        for i2 in 1:nvr
                            col = i2 + (j2 - 1) * nvr
                            alpha_ij += kh2.internal.SIG_H2_H2[row, col] * fH2[i2, j2, k]
                        end
                    end

                    s += kh2.dvr_vol[i] *
                         alpha_ij *
                         fH2[i, j, k] *
                         dvxj
                end
            end

            raw = Wperp_paraH2[k]
            mag = max(abs(raw), kh2.Wpp_tol)
            Wpp = raw >= 0.0 ? mag : -mag

            Omega_H2_H2[k] = max(s / (nH2[k] * Wpp), 0.0)
        end
    end
    return KH2CollisionType(Omega_H2_H2, Omega_H2_P, Omega_H2_H)
end

function _compute_collision_frequency(
    kh2::KineticH2,
    collision_freqs::KH2CollisionType{Vector{Float64}},
    gamma_wall::Array{Float64,3},
)::Array{Float64,3}
    size(gamma_wall) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("gamma_wall must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    length(collision_freqs.H2_H2) == kh2.nx || throw(DimensionMismatch("collision frequency lengths must match kh2.nx"))
    length(collision_freqs.H2_P) == kh2.nx || throw(DimensionMismatch("collision frequency lengths must match kh2.nx"))
    length(collision_freqs.H2_H) == kh2.nx || throw(DimensionMismatch("collision frequency lengths must match kh2.nx"))
    kh2.internal.Alpha_Loss === nothing && throw(ArgumentError("Alpha_Loss must be initialized before computing collision frequency"))

    nvr = kh2.nvr
    nvx = kh2.nvx
    nx = kh2.nx

    alpha_c = Array{Float64,3}(undef, nvr, nvx, nx)
    if kh2.collisions.H2_P_CX
        kh2.internal.Alpha_CX === nothing && throw(ArgumentError("Alpha_CX must be initialized when H2_P_CX is enabled"))
        @inbounds for k in 1:nx
            omega_el = collision_freqs.H2_P[k] + collision_freqs.H2_H[k] + collision_freqs.H2_H2[k]
            base = kh2.internal.Alpha_Loss[k] + omega_el
            for j in 1:nvx, i in 1:nvr
                alpha_c[i, j, k] = kh2.internal.Alpha_CX[i, j, k] + base + gamma_wall[i, j, k]
            end
        end
    else
        @inbounds for k in 1:nx
            omega_el = collision_freqs.H2_P[k] + collision_freqs.H2_H[k] + collision_freqs.H2_H2[k]
            base = kh2.internal.Alpha_Loss[k] + omega_el
            for j in 1:nvx, i in 1:nvr
                alpha_c[i, j, k] = base + gamma_wall[i, j, k]
            end
        end
    end

    _test_grid_spacing(kh2, alpha_c)
    return alpha_c
end

function _test_grid_spacing(kh2::KineticH2, alpha_c::Array{Float64,3})::Nothing
    size(alpha_c) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("alpha_c must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    _debrief_msg(kh2, "Testing x grid spacing", 1)

    max_dx_full = fill(1.0e32, kh2.nx)
    @inbounds for k in 1:kh2.nx
        for j in kh2.vx_pos
            vx2 = 2.0 * kh2.mesh.vx[j]
            for i in 1:kh2.nvr
                local_dx = vx2 / alpha_c[i, j, k]
                if isfinite(local_dx) && local_dx > 0.0 && local_dx < max_dx_full[k]
                    max_dx_full[k] = local_dx
                end
            end
        end
    end

    max_dx = Vector{Float64}(undef, max(kh2.nx - 1, 0))
    @inbounds for k in 1:length(max_dx)
        max_dx[k] = min(max_dx_full[k], max_dx_full[k + 1])
    end
    kh2.errors.Max_dx = max_dx

    @inbounds for k in 1:length(max_dx)
        dx = kh2.mesh.x[k + 1] - kh2.mesh.x[k]
        if max_dx[k] < dx
            throw(ErrorException("Kinetic_H2 => x mesh spacing is too large at interval $k"))
        end
    end
    return nothing
end

function _compute_mesh_equation_coefficients(
    kh2::KineticH2,
    alpha_c::Array{Float64,3},
    SH2::Vector{Float64},
)::KH2MeshEqCoefficients
    size(alpha_c) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("alpha_c must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    length(SH2) == kh2.nx || throw(DimensionMismatch("SH2 length must match kh2.nx"))
    kh2.internal.fw_hat === nothing && throw(ArgumentError("fw_hat must be initialized before computing mesh equation coefficients"))

    nvr = kh2.nvr
    nvx = kh2.nvx
    nx = kh2.nx

    Ak = zeros(Float64, nvr, nvx, nx)
    Bk = zeros(Float64, nvr, nvx, nx)
    Ck = zeros(Float64, nvr, nvx, nx)
    Dk = zeros(Float64, nvr, nvx, nx)
    Fk = zeros(Float64, nvr, nvx, nx)
    Gk = zeros(Float64, nvr, nvx, nx)

    @inbounds for k in 1:(nx - 1)
        kp1 = k + 1
        xdiff = kh2.mesh.x[kp1] - kh2.mesh.x[k]
        sh2_pair = SH2[kp1] + SH2[k]

        for j in kh2.vx_pos
            vxj = kh2.mesh.vx[j]
            for i in 1:nvr
                denom = 2.0 * vxj + xdiff * alpha_c[i, j, kp1]
                Ak[i, j, k] = (2.0 * vxj - xdiff * alpha_c[i, j, k]) / denom
                Bk[i, j, k] = xdiff / denom
                Fk[i, j, k] = xdiff * kh2.internal.fw_hat[i, j] * sh2_pair / (kh2.vth * denom)
            end
        end

        for j in kh2.vx_neg
            vxj = kh2.mesh.vx[j]
            for i in 1:nvr
                denom = -2.0 * vxj + xdiff * alpha_c[i, j, k]
                Ck[i, j, kp1] = (-2.0 * vxj - xdiff * alpha_c[i, j, kp1]) / denom
                Dk[i, j, kp1] = xdiff / denom
                Gk[i, j, kp1] = xdiff * kh2.internal.fw_hat[i, j] * sh2_pair / (kh2.vth * denom)
            end
        end
    end

    return KH2MeshEqCoefficients(Ak, Bk, Ck, Dk, Fk, Gk)
end


function _run_generations(kh2::KineticH2, 
    fH2::Array{Float64,3}, 
    nH2::Vector{Float64},
    fH2G::Array{Float64,3},
    NH2G::Matrix{Float64},
    nHP::Vector{Float64},
    gamma_wall::Array{Float64,3},
    meq_coeffs::KH2MeshEqCoefficients,
    collision_freqs::KH2CollisionType{Vector{Float64}},
    fH2_iterate::Bool)::Tuple{Array{Float64,3},
    Vector{Float64},
    Array{Float64,3},
    Matrix{Float64},
    Array{Float64,3},
    Array{Float64,3},
    KH2CollisionType{Array{Float64,3}},
    Int
}
    size(fH2) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("fH2 must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    length(nH2) == kh2.nx || throw(DimensionMismatch("nH2 length must match kh2.nx"))
    size(fH2G) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("fH2G must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    size(NH2G, 1) == kh2.nx || throw(DimensionMismatch("NH2G must have kh2.nx rows"))
    length(nHP) == kh2.nx || throw(DimensionMismatch("nHP length must match kh2.nx"))
    size(gamma_wall) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("gamma_wall must have size (kh2.nvr, kh2.nvx, kh2.nx)"))

    nvr = kh2.nvr
    nvx = kh2.nvx
    nx = kh2.nx

    Beta_CX_sum = zeros(Float64, nvr, nvx, nx)
    Swall_sum = zeros(Float64, nvr, nvx, nx)
    m_sums = KH2CollisionType(
        zeros(Float64, nvr, nvx, nx),
        zeros(Float64, nvr, nvx, nx),
        zeros(Float64, nvr, nvx, nx),
    )
    
    igen = 0
    while true
        if igen >= kh2.max_gen
            error(
                "Kinetic_H2: failed to converge after $(kh2.max_gen) generations. " *
                "The $(kh2.max_gen)th generation is still contributing a non-negligible amount " *
                "to the total neutral density. This means there are neutrals undergoing " *
                "$(kh2.max_gen) charge exchange or scattering events before ionisation, which " *
                "is unlikely in typical tokamak conditions and probably indicates a problem " *
                "with the input profiles."
            )
        end
        if !fH2_iterate
            break
        end
        igen += 1
        _debrief_msg(kh2, "Running generation $igen", 1)

        Swall = _compute_swall(kh2, fH2G, gamma_wall)
        @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
            Swall_sum[i, j, k] += Swall[i, j, k]
        end

        Beta_CX = _compute_beta_cx(kh2, fH2G, nHP)
        @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr 
            Beta_CX_sum[i, j, k] += Beta_CX[i, j, k]
        end

        m_vals = _compute_mh_values(kh2, fH2G, view(NH2G, :, igen))

        @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
            m_sums.H2_H2[i, j, k] += m_vals.H2_H2[i, j, k]
            m_sums.H2_P[i, j, k] += m_vals.H2_P[i, j, k]
            m_sums.H2_H[i, j, k] += m_vals.H2_H[i, j, k]
        end

        OmegaM = Array{Float64, 3}(undef, nvr, nvx, nx)
        @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
            OmegaM[i, j, k] = collision_freqs.H2_H2[k] * m_vals.H2_H2[i, j, k] +
                              collision_freqs.H2_P[k] * m_vals.H2_P[i, j, k] +
                              collision_freqs.H2_H[k] * m_vals.H2_H[i, j, k]
        end

        # Build the next generation by transporting the new source terms.
        fill!(fH2G, 0.0)
        @inbounds for k in 1:(nx - 1)
            kp1 = k + 1 
            for j in kh2.vx_pos
                for i in 1:nvr
                    fH2G[i, j, kp1] = 
                        meq_coeffs.A[i, j, k] * fH2G[i, j, k] +
                        meq_coeffs.B[i, j, k] *
                        (
                            Swall[i, j, kp1] + Beta_CX[i, j, kp1] + OmegaM[i, j, kp1] +
                            Swall[i, j, k]   + Beta_CX[i, j, k]   + OmegaM[i, j, k]
                        )
                end
            end
        end

        #Backward Sweep
        @inbounds for k in nx:-1:2
            km1 = k - 1

            for j in kh2.vx_neg
                for i in 1:nvr
                    fH2G[i, j, km1] =
                        meq_coeffs.C[i, j, k] * fH2G[i, j, k] +
                        meq_coeffs.D[i, j, k] *
                        (
                            Swall[i, j, km1] + Beta_CX[i, j, km1] + OmegaM[i, j, km1] +
                            Swall[i, j, k]   + Beta_CX[i, j, k]   + OmegaM[i, j, k]
                        )
                end
            end
        end

        @inbounds for k in 1:nx
            s = 0.0
            for j in 1:nvx
                dvxj = kh2.dvx[j]
                for i in 1:nvr
                    s += kh2.dvr_vol[i] * fH2G[i, j, k] * dvxj
                end
            end
            NH2G[k, igen + 1] = s
        end

        # Add this generation to the total neutral distribution and density.
        @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
            fH2[i, j, k] += fH2G[i, j, k]
        end

        @inbounds for k in 1:nx
            nH2[k] += NH2G[k, igen + 1]
        end

        max_nH2 = maximum(nH2)
        Delta_nH2G = maximum(@view NH2G[:, igen + 1]) / max_nH2

        # Stop when the newest generation is small relative to the total density.
        # During outer fixed-point iteration, do not over-solve the inner series
        # far beyond the current outer density error.
        if (Delta_nH2G < kh2.truncate) ||
           (fH2_iterate && Delta_nH2G < 0.003 * kh2.internal.Delta_nH2s)
            break
        end
    end

    return fH2, nH2, fH2G, NH2G, Swall_sum, Beta_CX_sum, m_sums, igen
end

function _compute_iteration_results(
    kh2::KineticH2,
    fH2::Array{Float64,3},
)::Tuple{
    Vector{Float64},
    Vector{Float64},
    Vector{Float64},
    Array{Float64,3},
    Vector{Float64},
    Vector{Float64},
    Vector{Float64},
    Vector{Float64},
    Vector{Float64},
    Vector{Float64},
}
    size(fH2) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("fH2 must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    kh2.internal.sigv === nothing && throw(ArgumentError("sigv must be initialized before computing iteration results"))

    nvr = kh2.nvr
    nvx = kh2.nvx
    nx = kh2.nx

    nH2 = zeros(Float64, nx)
    GammaxH2 = zeros(Float64, nx)
    VxH2 = zeros(Float64, nx)
    pH2 = zeros(Float64, nx)
    TH2 = zeros(Float64, nx)
    vr2vx2_ran = Array{Float64,3}(undef, nvr, nvx, nx)

    @inbounds for k in 1:nx
        n_sum = 0.0
        flux_sum = 0.0
        for j in 1:nvx
            vxj = kh2.mesh.vx[j]
            dvxj = kh2.dvx[j]
            for i in 1:nvr
                w = kh2.dvr_vol[i] * dvxj
                f = fH2[i, j, k]
                n_sum += f * w
                flux_sum += f * vxj * w
            end
        end
        nH2[k] = n_sum
        GammaxH2[k] = kh2.vth * flux_sum
        VxH2[k] = GammaxH2[k] / n_sum
    end

    pH2_coef = (2.0 * kh2.mu * H_MASS) * kh2.vth^2 / (3.0 * Q)
    @inbounds for k in 1:nx
        vx_shift = VxH2[k] / kh2.vth
        p_sum = 0.0
        for j in 1:nvx
            vxran2 = (kh2.mesh.vx[j] - vx_shift)^2
            dvxj = kh2.dvx[j]
            for i in 1:nvr
                v2 = kh2.mesh.vr[i]^2 + vxran2
                vr2vx2_ran[i, j, k] = v2
                p_sum += kh2.dvr_vol[i] * v2 * fH2[i, j, k] * dvxj
            end
        end
        pH2[k] = pH2_coef * p_sum
        TH2[k] = pH2[k] / nH2[k]
    end

    NuDis = Vector{Float64}(undef, nx)
    NuE = Vector{Float64}(undef, nx)
    nHP = Vector{Float64}(undef, nx)
    THP = Vector{Float64}(undef, nx)
    @inbounds for k in 1:nx
        NuDis[k] = kh2.mesh.ne[k] * sum(@view kh2.internal.sigv[k, 8:11])
        NuE[k] = (7.7e-7 * kh2.mesh.ne[k] * 1.0e-6) / (sqrt(kh2.mu) * kh2.mesh.Ti[k]^1.5)
        nHP[k] = (nH2[k] * kh2.mesh.ne[k] * kh2.internal.sigv[k, 2]) / (NuDis[k] + kh2.NuLoss[k])
        THP[k] = (kh2.mesh.Ti[k] * NuE[k]) / (NuE[k] + NuDis[k] + kh2.NuLoss[k])
    end

    return nH2, GammaxH2, VxH2, vr2vx2_ran, pH2, TH2, NuDis, NuE, nHP, THP
end

# Returns fH2, alpha_c, Beta_CX_sum, Swall_sum, collision_freqs, m_sums.
function _run_iteration_scheme(
    kh2::KineticH2,
    fH2::Array{Float64,3},
    nH2::Vector{Float64},
    nHP::Vector{Float64},
    THP::Vector{Float64},
    SH2::Vector{Float64},
    gamma_wall::Array{Float64,3},
    Do_Alpha_CX::Bool,
    Do_Alpha_H2_P::Bool,
)::Tuple{
    Array{Float64,3},
    Array{Float64,3},
    Array{Float64,3},
    Array{Float64,3},
    KH2CollisionType{Vector{Float64}},
    KH2CollisionType{Array{Float64,3}},
}
    size(fH2) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("fH2 must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    length(nH2) == kh2.nx || throw(DimensionMismatch("nH2 length must match kh2.nx"))
    length(nHP) == kh2.nx || throw(DimensionMismatch("nHP length must match kh2.nx"))
    length(THP) == kh2.nx || throw(DimensionMismatch("THP length must match kh2.nx"))
    length(SH2) == kh2.nx || throw(DimensionMismatch("SH2 length must match kh2.nx"))
    size(gamma_wall) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("gamma_wall must have size (kh2.nvr, kh2.nvx, kh2.nx)"))

    nvr = kh2.nvr
    nvx = kh2.nvx
    nx = kh2.nx

    fH2_iterate = false
    if kh2.collisions.H2_H2_EL || kh2.collisions.H2_P_CX || kh2.collisions.H2_P_EL || kh2.collisions.H2_H_EL || kh2.ni_correct
        fH2_iterate = true
    end
    fH2G = zeros(Float64, nvr, nvx, nx)
    NH2G = zeros(Float64, nx, kh2.max_gen + 1)
    alpha_c = zeros(Float64, nvr, nvx, nx)
    Swall_sum = zeros(Float64, nvr, nvx, nx)
    Beta_CX_sum = zeros(Float64, nvr, nvx, nx)
    collision_freqs = KH2CollisionType(zeros(Float64, nx), zeros(Float64, nx), zeros(Float64, nx))
    m_sums = KH2CollisionType(
        zeros(Float64, nvr, nvx, nx),
        zeros(Float64, nvr, nvx, nx),
        zeros(Float64, nvr, nvx, nx),
    )
    igen = 0

    while true
        nH_input = copy(nH2)

        if Do_Alpha_CX
            _compute_alpha_cx!(kh2, nHP, THP)
        end
        if Do_Alpha_H2_P
            _compute_alpha_h2_p!(kh2, nHP)
        end

        collision_freqs = _compute_omega_values(kh2, fH2, nH2)
        alpha_c = _compute_collision_frequency(kh2, collision_freqs, gamma_wall)
        meq_coeffs = _compute_mesh_equation_coefficients(kh2, alpha_c, SH2)

        _debrief_msg(kh2, "Computing molecular neutral generation#0", 0)
        fill!(fH2G, 0.0)
        for j in kh2.vx_pos
            for i in 1:nvr
                fH2G[i, j, 1] = fH2[i, j, 1]
            end
        end

        # positive vx sweep: Python k=0:nx-2 -> Julia k=1:nx-1
        for k in 1:(nx - 1)
            kp1 = k + 1
            for j in kh2.vx_pos
                for i in 1:nvr
                    fH2G[i, j, kp1] =
                        fH2G[i, j, k] * meq_coeffs.A[i, j, k] +
                        meq_coeffs.F[i, j, k]
                end
            end
        end

        for k in nx:-1:2
            km1 = k - 1
            for j in kh2.vx_neg
                for i in 1:nvr
                    fH2G[i, j, km1] =
                        fH2G[i, j, k] * meq_coeffs.C[i, j, k] +
                        meq_coeffs.G[i, j, k]
                end
            end
        end

        for k in 1:nx
            s = 0.0
            for j in 1:nvx
                dvxj = kh2.dvx[j]
                for i in 1:nvr
                    s += kh2.dvr_vol[i] * fH2G[i, j, k] * dvxj
                end
            end
            NH2G[k, 1] = s
        end

        fH2 = copy(fH2G)
        nH2 = copy(NH2G[:, 1])

        fH2, nH2, fH2G, NH2G, Swall_sum, Beta_CX_sum, m_sums, igen = _run_generations(kh2, fH2, nH2, fH2G, NH2G, nHP, gamma_wall, meq_coeffs, collision_freqs, fH2_iterate)
        kh2.internal.MH2_H2_sum = m_sums.H2_H2

        nH2, _, _, _, _, _, _, _, nHP, THP = _compute_iteration_results(kh2, fH2)

        if fH2_iterate
            maxdiff = 0.0
            maxnH2 = 0.0
            @inbounds for k in 1:nx
                maxdiff = max(maxdiff, abs(nH_input[k] - nH2[k]))
                maxnH2 = max(maxnH2, nH2[k])
            end
            kh2.internal.Delta_nH2s = maxdiff / maxnH2
            if kh2.internal.Delta_nH2s <= 10.0 * kh2.truncate
                break
            end
        else
            break
        end
    end

    Swall = _compute_swall(kh2, fH2G, gamma_wall)
    Beta_CX = _compute_beta_cx(kh2, fH2G, nHP)
    m_vals = _compute_mh_values(kh2, fH2G, view(NH2G, :, igen + 1))

    @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
        Swall_sum[i, j, k] += Swall[i, j, k]
        Beta_CX_sum[i, j, k] += Beta_CX[i, j, k]
        m_sums.H2_H2[i, j, k] += m_vals.H2_H2[i, j, k]
        m_sums.H2_P[i, j, k] += m_vals.H2_P[i, j, k]
        m_sums.H2_H[i, j, k] += m_vals.H2_H[i, j, k]
    end

    return fH2, alpha_c, Beta_CX_sum, Swall_sum, collision_freqs, m_sums
end

function _compile_results(kh2::KineticH2,
    fH2::Array{Float64,3},
    SH2::Vector{Float64},
    gamma_wall::Array{Float64,3},
    alpha_c::Array{Float64,3},
    Beta_CX_sum::Array{Float64,3},
    Swall_sum::Array{Float64,3},
    collision_freqs::KH2CollisionType{Vector{Float64}},
    m_sums::KH2CollisionType{Array{Float64,3}})::KH2Results
    size(fH2) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("fH2 must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    length(SH2) == kh2.nx || throw(DimensionMismatch("SH2 length must match kh2.nx"))
    size(gamma_wall) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("gamma_wall must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    size(alpha_c) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("alpha_c must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    size(Beta_CX_sum) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("Beta_CX_sum must have size (kh2.nvr, kh2.nvx, kh2.nx)"))
    size(Swall_sum) == (kh2.nvr, kh2.nvx, kh2.nx) || throw(DimensionMismatch("Swall_sum must have size (kh2.nvr, kh2.nvx, kh2.nx)"))

    nvr, nvx, nx = kh2.nvr, kh2.nvx, kh2.nx
    vr, vx = kh2.mesh.vr, kh2.mesh.vx
    fw_hat = kh2.internal.fw_hat === nothing ? zeros(Float64, nvr, nvx) : kh2.internal.fw_hat
    kh2.internal.vr2vx2 === nothing && throw(ArgumentError("vr2vx2 must be initialized before compiling KineticH2 results"))
    kh2.internal.Alpha_Loss === nothing && throw(ArgumentError("Alpha_Loss must be initialized before compiling KineticH2 results"))

    nH2, GammaxH2, VxH2, vr2vx2_ran, pH2, TH2, NuDis, NuE, nHP, THP =
        _compute_iteration_results(kh2, fH2)

    VxH2_vth = Vector{Float64}(undef, nx)
    @inbounds for k in 1:nx
        VxH2_vth[k] = VxH2[k] / kh2.vth
    end

    for out in (
        kh2.output.piH2_xx, kh2.output.piH2_yy, kh2.output.piH2_zz,
        kh2.output.RxH2CX, kh2.output.RxH_H2, kh2.output.RxP_H2, kh2.output.RxW_H2,
        kh2.output.EH2CX, kh2.output.EH_H2, kh2.output.EP_H2, kh2.output.EW_H2,
        kh2.output.Epara_PerpH2_H2,
    )
        fill!(out, 0.0)
    end

    piH_coef = 2.0 * kh2.mu * H_MASS * kh2.vth^2 / Q
    qxH2_coef = 0.5 * (2.0 * kh2.mu * H_MASS) * kh2.vth^3
    E_coef = 0.5 * (2.0 * kh2.mu * H_MASS) * kh2.vth^2
    Rx_coef = (2.0 * kh2.mu * H_MASS) * kh2.vth

    qxH2 = zeros(Float64, nx)
    qxH2_total = zeros(Float64, nx)
    Sloss = zeros(Float64, nx)
    QH2 = zeros(Float64, nx)
    RxH2 = zeros(Float64, nx)
    QH2_total = zeros(Float64, nx)
    WallH2 = zeros(Float64, nx)

    @inbounds for k in 1:nx
        pi_xx_sum = 0.0
        pi_yy_sum = 0.0
        qx_sum = 0.0

        for j in 1:nvx
            vxran = vx[j] - VxH2_vth[k]
            vxran2 = vxran * vxran
            dvxj = kh2.dvx[j]
            for i in 1:nvr
                vr2 = vr[i] * vr[i]
                f = fH2[i, j, k]
                w = kh2.dvr_vol[i]
                pi_xx_sum += w * f * vxran2 * dvxj
                pi_yy_sum += w * vr2 * f * dvxj
                qx_sum += w * vr2vx2_ran[i, j, k] * f * vxran * dvxj
            end
        end

        kh2.output.piH2_xx[k] = piH_coef * pi_xx_sum - pH2[k]
        kh2.output.piH2_yy[k] = 0.5 * piH_coef * pi_yy_sum - pH2[k]
        kh2.output.piH2_zz[k] = kh2.output.piH2_yy[k]
        qxH2[k] = qxH2_coef * qx_sum
    end

    @inbounds for k in 1:nx
        Q_sum = 0.0
        Rx_sum = 0.0
        C_sum = 0.0
        wall_sum = 0.0
        RxCX_sum = 0.0
        ECX_sum = 0.0
        RxH_sum = 0.0
        EH_sum = 0.0
        RxP_sum = 0.0
        EP_sum = 0.0
        RxW_sum = 0.0
        EW_sum = 0.0
        Epara_sum = 0.0

        for j in 1:nvx
            vxran = vx[j] - VxH2_vth[k]
            dvxj = kh2.dvx[j]
            for i in 1:nvr
                f = fH2[i, j, k]
                w = kh2.dvr_vol[i]
                vr2vx2 = kh2.internal.vr2vx2[i, j, k]
                C = kh2.vth * (
                    fw_hat[i, j] * SH2[k] / kh2.vth +
                    Swall_sum[i, j, k] +
                    Beta_CX_sum[i, j, k] -
                    alpha_c[i, j, k] * f +
                    collision_freqs.H2_P[k] * m_sums.H2_P[i, j, k] +
                    collision_freqs.H2_H[k] * m_sums.H2_H[i, j, k] +
                    collision_freqs.H2_H2[k] * m_sums.H2_H2[i, j, k]
                )

                Q_sum += w * vr2vx2_ran[i, j, k] * C * dvxj
                Rx_sum += w * C * vxran * dvxj
                C_sum += w * C * dvxj
                wall_sum += w * gamma_wall[i, j, k] * f * dvxj

                if kh2.collisions.H2_P_CX
                    CCX = kh2.vth * (Beta_CX_sum[i, j, k] - kh2.internal.Alpha_CX[i, j, k] * f)
                    RxCX_sum += w * CCX * vxran * dvxj
                    ECX_sum += w * vr2vx2 * CCX * dvxj
                end
                if kh2.collisions.H2_H_EL
                    CH2_H = kh2.vth * collision_freqs.H2_H[k] * (m_sums.H2_H[i, j, k] - f)
                    RxH_sum += w * CH2_H * vxran * dvxj
                    EH_sum += w * vr2vx2 * CH2_H * dvxj
                end
                if kh2.collisions.H2_P_EL
                    CH2_P = kh2.vth * collision_freqs.H2_P[k] * (m_sums.H2_P[i, j, k] - f)
                    RxP_sum += w * CH2_P * vxran * dvxj
                    EP_sum += w * vr2vx2 * CH2_P * dvxj
                end

                CW_H2 = kh2.vth * (Swall_sum[i, j, k] - gamma_wall[i, j, k] * f)
                RxW_sum += w * CW_H2 * vxran * dvxj
                EW_sum += w * vr2vx2 * CW_H2 * dvxj

                if kh2.collisions.H2_H2_EL
                    CH2_H2 = kh2.vth * collision_freqs.H2_H2[k] * (m_sums.H2_H2[i, j, k] - f)
                    vr2_2vx_ran2 = vr[i]^2 - 2.0 * vxran^2
                    Epara_sum += w * vr2_2vx_ran2 * CH2_H2 * dvxj
                end
            end
        end

        QH2[k] = E_coef * Q_sum
        RxH2[k] = Rx_coef * Rx_sum
        Sloss[k] = -C_sum + SH2[k]
        WallH2[k] = wall_sum
        kh2.output.RxH2CX[k] = Rx_coef * RxCX_sum
        kh2.output.EH2CX[k] = E_coef * ECX_sum
        kh2.output.RxH_H2[k] = Rx_coef * RxH_sum
        kh2.output.EH_H2[k] = E_coef * EH_sum
        kh2.output.RxP_H2[k] = Rx_coef * RxP_sum
        kh2.output.EP_H2[k] = E_coef * EP_sum
        kh2.output.RxW_H2[k] = Rx_coef * RxW_sum
        kh2.output.EW_H2[k] = E_coef * EW_sum
        kh2.output.Epara_PerpH2_H2[k] = -E_coef * Epara_sum
    end

    @inbounds for k in 1:nx
        qxH2_total[k] =
            (0.5 * nH2[k] * (2.0 * kh2.mu * H_MASS) * VxH2[k]^2 + 2.5 * pH2[k] * Q) * VxH2[k] +
            Q * kh2.output.piH2_xx[k] * VxH2[k] +
            qxH2[k]
        QH2_total[k] =
            QH2[k] +
            RxH2[k] * VxH2[k] -
            0.5 * (2.0 * kh2.mu * H_MASS) * (Sloss[k] - SH2[k]) * VxH2[k]^2
    end

    gammax_plus = 0.0
    gammax_minus = 0.0
    @inbounds begin
        for j in kh2.vx_pos
            vx_dvx = vx[j] * kh2.dvx[j]
            for i in 1:nvr
                gammax_plus += kh2.dvr_vol[i] * fH2[i, j, 1] * vx_dvx
            end
        end
        for j in kh2.vx_neg
            vx_dvx = vx[j] * kh2.dvx[j]
            for i in 1:nvr
                gammax_minus += kh2.dvr_vol[i] * fH2[i, j, 1] * vx_dvx
            end
        end
    end
    gammax_plus *= kh2.vth
    gammax_minus *= kh2.vth
    AlbedoH2 = abs(gammax_plus) > 0.0 ? -gammax_minus / gammax_plus : 0.0

    fSH, SH, SP, SHP, ESH = _compute_h_source(kh2, nH2, nHP, THP, SH2, TH2, GammaxH2)

    return KH2Results(
        fH2, nHP, THP, nH2, GammaxH2, VxH2, pH2, TH2, qxH2, qxH2_total,
        Sloss, QH2, RxH2, QH2_total, AlbedoH2, WallH2, fSH, SH, SP, SHP,
        NuE, NuDis, ESH, kh2.Eaxis,
    )
end

function _compute_h_source(
    kh2::KineticH2,
    nH2::Vector{Float64},
    nHP::Vector{Float64},
    THP::Vector{Float64},
    SH2::Vector{Float64},
    TH2::Vector{Float64},
    GammaxH2::Vector{Float64},
)::Tuple{Array{Float64,3},Vector{Float64},Vector{Float64},Vector{Float64},Matrix{Float64}}
    nvr = kh2.nvr
    nvx = kh2.nvx
    nx  = kh2.nx

    fSH = zeros(Float64, nvr, nvx, nx)
    SH  = zeros(Float64, nx)
    SP  = zeros(Float64, nx)
    SHP = zeros(Float64, nx)
    ESH = zeros(Float64, nvr, nx)

    if !kh2.compute_h_source
        return fSH, SH, SP, SHP, ESH
    end

    _debrief_msg(kh2, "Computing Velocity Distributions of H products...", 1)

    # Reactions: R2, R3, R4, R5, R6, R7, R8, R10
    SFCn = zeros(Float64, nvr, nvx, nx, 8)
    Vfc  = zeros(Float64, nvr, nvx, nx)
    Tfc  = zeros(Float64, nvr, nvx, nx)

    kh2.internal.vr2vx2 === nothing && throw(ArgumentError("vr2vx2 must be initialized before computing H source"))
    kh2.internal.sigv === nothing && throw(ArgumentError("sigv must be initialized before computing H source"))

    magV = sqrt.(kh2.internal.vr2vx2)

    # Lookup table
    nFC, Eave, Emax, Emin = _generate_h_source_table(kh2)

    _THP = THP ./ kh2.mesh.Tnorm
    _TH2 = TH2 ./ kh2.mesh.Tnorm

    reactions = (2, 3, 4, 5, 6, 7, 8, 10)

    for reaction in reactions
        ii = nFC[reaction]

        @views begin
            Tfc[1, 1, :] .= 0.25 .* (Emax[:, ii] .- Emin[:, ii]) ./ kh2.mesh.Tnorm
            Vfc[1, 1, :] .= sqrt.(Eave[:, ii] ./ kh2.mesh.Tnorm)
        end

        @inbounds for k in 1:nx
            Vfc[:, :, k] .= Vfc[1, 1, k]
            Tfc[:, :, k] .= Tfc[1, 1, k]
        end

        if reaction <= 6
            @inbounds for k in 1:nx
                denom = Tfc[1, 1, k] + 0.5 * _TH2[k]

                @views arg = .-(
                    magV[:, :, k] .- Vfc[:, :, k] .+ 1.5 .* Tfc[:, :, k] ./ Vfc[:, :, k]
                ).^2 ./ denom

                @views SFCn[:, :, k, ii] .= exp.(max.(arg, -80.0))
            end
        else
            @inbounds for k in 1:nx
                denom = Tfc[1, 1, k] + 0.5 * _THP[k]

                @views arg = .-(
                    magV[:, :, k] .- Vfc[:, :, k] .+ 1.5 .* Tfc[:, :, k] ./ Vfc[:, :, k]
                ).^2 ./ denom

                @views SFCn[:, :, k, ii] .= exp.(max.(arg, -80.0))
            end
        end

        # Normalize source distribution at each x-location
        @inbounds for k in 1:nx
            @views norm_val = sum(kh2.dvr_vol .* (SFCn[:, :, k, ii] * kh2.dvx))
            @views SFCn[:, :, k, ii] ./= norm_val
        end
    end

    if kh2.compute_errors
        kh2.errors.vbar_error = 0.0

        Emin_global = minimum(Eave[1, :])
        Emax_global = maximum(Eave[1, :])

        TFC = Emin_global .+ (Emax_global - Emin_global) .* collect(0:nx-1) ./ (nx - 1)
        vx_shift = zeros(Float64, nx)

        Maxwell = create_shifted_maxwellian(
            kh2.mesh.vr,
            kh2.mesh.vx,
            TFC,
            vx_shift,
            kh2.mu,
            1,
            kh2.mesh.Tnorm,
        )

        @views vbar_test = kh2.vth .* sqrt.(kh2.internal.vr2vx2[:, :, 1])

        @inbounds for k in 1:nx
            @views vbar = sum(kh2.dvr_vol .* ((vbar_test .* Maxwell[:, :, k]) * kh2.dvx))
            vbar_exact = 2 * kh2.vth * sqrt(TFC[k] / kh2.mesh.Tnorm) / sqrt(pi)
            kh2.errors.vbar_error = max(kh2.errors.vbar_error, abs(vbar - vbar_exact) / vbar_exact)
        end

        _debrief_msg(
            kh2,
            "Maximum Vbar error over FC energy range = " *
            sval(kh2.errors.vbar_error),
            0,
        )
    end

    # Compute atomic hydrogen source distribution function
    SH_coef = kh2.mesh.ne .* nH2

    # Python stores reaction Rn in sigv[:, n] with a dummy column 0. Julia keeps
    # the dummy column as column 1, so reaction Rn is column n+1.
    @inline sigv_reaction(k, reaction) = kh2.internal.sigv[k, reaction + 1]
    @inline fSH_calc(k, reaction) = sigv_reaction(k, reaction) .* @view(SFCn[:, :, k, nFC[reaction]])

    @inbounds for k in 1:nx
        @views fSH[:, :, k] .= SH_coef[k] .* (
            2 .* fSH_calc(k, 2) .+
            2 .* fSH_calc(k, 3) .+
                fSH_calc(k, 4) .+
            2 .* fSH_calc(k, 5) .+
            2 .* fSH_calc(k, 6)
        )

        @views fSH[:, :, k] .+= kh2.mesh.ne[k] * nHP[k] .* (
                fSH_calc(k, 7) .+
                fSH_calc(k, 8) .+
            2 .* fSH_calc(k, 10)
        )
    end

    # Compute total H and H(+) sources
    @inbounds for k in 1:nx
        @views SH[k] = sum(kh2.dvr_vol .* (fSH[:, :, k] * kh2.dvx))

        SP[k] =
            SH_coef[k] * sigv_reaction(k, 4) +
            kh2.mesh.ne[k] * nHP[k] * (
                sigv_reaction(k, 7) +
                sigv_reaction(k, 8) +
                2 * sigv_reaction(k, 9)
            )
    end

    # Compute total HP source
    @inbounds for k in 1:nx
        SHP[k] = SH_coef[k] * sigv_reaction(k, 1)
    end

    # Compute energy distribution of H source
    @inbounds for k in 1:nx
        @views ESH[:, k] .= (
            kh2.Eaxis .* fSH[:, kh2.vx_pos[1], k] .* kh2.dvr_vol_h_order
        ) ./ kh2.dEaxis

        max_esh = maximum(@view ESH[:, k])
        if max_esh != 0.0
            @views ESH[:, k] ./= max_esh
        end
    end

    # Compute Source Error
    if kh2.compute_errors
        Source_Error = zeros(Float64, nx)

        _debrief_msg(kh2, "Computing Source Error", 1)

        dGammaxH2dx = zeros(Float64, nx - 1)
        SH_p        = zeros(Float64, nx - 1)

        @inbounds for k in 1:nx-1
            dGammaxH2dx[k] =
                (GammaxH2[k + 1] - GammaxH2[k]) /
                (kh2.mesh.x[k + 1] - kh2.mesh.x[k])
        end

        @inbounds for k in 1:nx-1
            SH_p[k] = 0.5 * (
                SH[k + 1] + SP[k + 1] + 2 * kh2.NuLoss[k + 1] * nHP[k + 1] - 2 * SH2[k + 1] +
                SH[k]     + SP[k]     + 2 * kh2.NuLoss[k]     * nHP[k]     - 2 * SH2[k]
            )
        end

        max_source = maximum(vcat(SH, 2 .* SH2))

        @inbounds for k in 1:nx-1
            denom = maximum(abs.([2 * dGammaxH2dx[k], SH_p[k], max_source]))
            Source_Error[k] = abs(2 * dGammaxH2dx[k] + SH_p[k]) / denom
        end

        _debrief_msg(kh2, "Maximum Normalized Source_error = " * sval(maximum(Source_Error)), 0)
    end

    return fSH, SH, SP, SHP, ESH
end

function _generate_h_source_table(kh2::KineticH2)
    nx = kh2.nx

    # Julia version of Python:
    # nFC = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 7])
    #
    # In Julia we use reaction number directly:
    # nFC[2]  -> R2 index
    # nFC[10] -> R10 index
    #
    # But because Julia is 1-based, FC bins are 1:8 instead of 0:7.
    nFC = zeros(Int, 10)
    nFC[2]  = 1
    nFC[3]  = 2
    nFC[4]  = 3
    nFC[5]  = 4
    nFC[6]  = 5
    nFC[7]  = 6
    nFC[8]  = 7
    nFC[10] = 8

    Eave = zeros(Float64, nx, 8)
    Emax = zeros(Float64, nx, 8)
    Emin = zeros(Float64, nx, 8)

    # Reaction R2: e + H2 -> e + H(1s) + H(1s)
    ii = nFC[2]
    Eave[:, ii] .= 3.0
    Emax[:, ii] .= 4.25
    Emin[:, ii] .= 2.0

    # Reaction R3: e + H2 -> e + H(1s) + H*(2s)
    ii = nFC[3]
    Eave[:, ii] .= 0.3
    Emax[:, ii] .= 0.55
    Emin[:, ii] .= 0.0

    # Reaction R4: e + H2 -> e + H(+) + H(1s) + e
    ii = nFC[4]

    Ee = 1.5 .* kh2.mesh.Te

    @inbounds for k in 1:nx
        if Ee[k] <= 26.0
            Eave[k, ii] = 0.25
        elseif Ee[k] <= 41.6
            Eave[k, ii] = max(0.5 * (Ee[k] - 26.0), 0.25)
        else
            Eave[k, ii] = 7.8
        end
    end

    Emax[:, ii] .= 1.5 .* Eave[:, ii]
    Emin[:, ii] .= 0.5 .* Eave[:, ii]

    # Reaction R5: e + H2 -> e + H*(2p) + H*(2s)
    ii = nFC[5]
    Eave[:, ii] .= 4.85
    Emax[:, ii] .= 5.85
    Emin[:, ii] .= 2.85

    # Reaction R6: e + H2 -> e + H(1s) + H*(n=3)
    ii = nFC[6]
    Eave[:, ii] .= 2.5
    Emax[:, ii] .= 3.75
    Emin[:, ii] .= 1.25

    # Reaction R7: e + H2(+) -> e + H(+) + H(1s)
    ii = nFC[7]
    Eave[:, ii] .= 4.3
    Emax[:, ii] .= 4.3 + 2.1
    Emin[:, ii] .= 4.3 - 2.1

    # Reaction R8: e + H2(+) -> e + H(+) + H*(n=2)
    ii = nFC[8]
    Eave[:, ii] .= 1.5
    Emax[:, ii] .= 1.5 + 0.75
    Emin[:, ii] .= 1.5 - 0.75

    # Reaction R10: e + H2(+) -> H(1s) + H*(n>=2)
    ii = nFC[10]

    # Relative cross sections for n = 2, 3, 4, 5, 6
    R10rel = Float64[0.1, 0.45, 0.22, 0.12, 0.069]

    for k in 7:10
        push!(R10rel, 10.0 / k^3)
    end

    # Energy levels for n = 2:10
    En = 13.58 ./ ((2:10) .^ 2)

    truncate_point = min(length(Ee), length(En))

    EHn = 0.5 .* (Ee[1:truncate_point] .- En[1:truncate_point]) .*
          R10rel[1:truncate_point] ./ sum(R10rel)

    EHn .= max.(EHn, 0.0)

    R10_Eave = max(sum(EHn), 0.25)

    Eave[:, ii] .= R10_Eave
    Emax[:, ii] .= 1.5 .* Eave[:, ii]
    Emin[:, ii] .= 0.5 .* Eave[:, ii]

    return nFC, Eave, Emax, Emin
end
