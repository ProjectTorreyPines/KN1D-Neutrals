const KHArray3 = Array{Float64,3}
const KHArray5 = Array{Float64,5}
const KH_NTHETA = 5
const KH_DTHETA = fill(1.0 / KH_NTHETA, KH_NTHETA)
const KH_COS_THETA = cos.(π .* ((0:KH_NTHETA-1) .+ 0.5) ./ KH_NTHETA)

struct KHCollisions
    # Enabled collision channels for the atomic-H solve.
    H2_H_EL::Bool
    H_H_EL::Bool
    H_P_EL::Bool
    H_P_CX::Bool
    SIMPLE_CX::Bool
end

struct MeshEqCoefficients
    # Directional transport recurrence coefficients for the discretized steady
    # kinetic equation. A/B/F are used for vx > 0 sweeps; C/D/G for vx < 0.
    A::KHArray3
    B::KHArray3
    C::KHArray3
    D::KHArray3
    F::KHArray3
    G::KHArray3
end

struct CollisionType{T}
    # Same physical quantity grouped by collision partner: atomic H, proton,
    # and molecular H2. T is usually Vector{Float64} or Array{Float64,3}.
    H_H::T
    H_P::T
    H_H2::T
end

struct KHResults
    # Public result returned by run_procedure.
    fH::KHArray3
    # Final atomic neutral distribution fH(vr, vx, x).
    nH::Vector{Float64}
    # Atomic neutral density profile.
    GammaxH::Vector{Float64}
    # Atomic neutral particle flux in x.
    VxH::Vector{Float64}
    # Mean axial neutral velocity.
    pH::Vector{Float64}
    # Scalar neutral pressure.
    TH::Vector{Float64}
    # Neutral temperature.
    qxH::Vector{Float64}
    # Conductive/thermal heat flux.
    qxH_total::Vector{Float64}
    # Total heat flux including flow and pressure work terms.
    NetHSource::Vector{Float64}
    # Net volumetric H source from the final collision/source balance.
    Sion::Vector{Float64}
    # Ionization source rate from atomic H.
    QH::Vector{Float64}
    # Net thermal energy transfer into neutral atoms.
    RxH::Vector{Float64}
    # Net x-momentum transfer to neutral atoms.
    QH_total::Vector{Float64}
    # Total energy transfer including flow-energy terms.
    AlbedoH::Float64
    # Reflected/incoming flux ratio at the left boundary.
    SideWallH::Vector{Float64}
    # Atomic neutral loss rate to side walls.
end

mutable struct KineticHOutput
    # Additional diagnostic channels stored on the solver object to mirror the
    # Python Output dataclass.
    piH_xx::Vector{Float64}
    # xx component of neutral stress tensor minus scalar pressure.
    piH_yy::Vector{Float64}
    # yy component of neutral stress tensor minus scalar pressure.
    piH_zz::Vector{Float64}
    # zz component; same as yy by cylindrical symmetry.
    RxHCX::Vector{Float64}
    # Momentum exchange from charge exchange.
    RxH2_H::Vector{Float64}
    # Momentum exchange from H2-H elastic collisions.
    RxP_H::Vector{Float64}
    # Momentum exchange from proton-H elastic collisions.
    RxW_H::Vector{Float64}
    # Momentum loss to side walls.
    EHCX::Vector{Float64}
    # Energy exchange from charge exchange.
    EH2_H::Vector{Float64}
    # Energy exchange from H2-H elastic collisions.
    EP_H::Vector{Float64}
    # Energy exchange from proton-H elastic collisions.
    EW_H::Vector{Float64}
    # Energy loss to side walls.
    Epara_PerpH_H::Vector{Float64}
    # Parallel/perpendicular energy exchange diagnostic for H-H elastic collisions.
    SourceH::Vector{Float64}
    # User-supplied/direct atomic source integrated over velocity.
    SRecomb::Vector{Float64}
    # Recombination source profile.
end

function KineticHOutput(nx::Integer)::KineticHOutput
    n = Int(nx)
    return KineticHOutput((zeros(Float64, n) for _ in 1:14)...)
end

mutable struct KineticHErrors
    # Optional diagnostic/error arrays populated only when requested.
    Max_dx::Union{Nothing,Vector{Float64}}
    # Maximum stable x spacing implied by the local loss frequency.
    vbar_error::Union{Nothing,Float64}
    # Velocity-averaging consistency error.
    mesh_error::Union{Nothing,Vector{Float64}}
    # Mesh-equation residual/error diagnostic.
    moment_error::Union{Nothing,Vector{Float64}}
    # Moment-conservation error diagnostic.
    C_Error::Union{Nothing,Vector{Float64}}
    # Total collision/source balance error.
    CX_Error::Union{Nothing,Vector{Float64}}
    # Charge-exchange source/sink consistency error.
    H_H_error::Union{Nothing,Vector{Float64}}
    # H-H elastic collision consistency error.
    qxH_total_error::Union{Nothing,Vector{Float64}}
    # Heat-flux identity error.
    QH_total_error::Union{Nothing,Vector{Float64}}
    # Total-energy balance error.
end

KineticHErrors() = KineticHErrors(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)

mutable struct KineticHInput
    # Copies of the inputs used in the most recent run, for diagnostics and
    # parity with the Python Input object.
    fH2_s::Union{Nothing,KHArray3}
    # Molecular H2 distribution supplied to run_procedure.
    fSH_s::Union{Nothing,KHArray3}
    # User/direct atomic H source distribution.
    nHP_s::Union{Nothing,Vector{Float64}}
    # Extra proton density contribution from coupled molecular processes.
    THP_s::Union{Nothing,Vector{Float64}}
    # Temperature associated with nHP_s.
    fH_s::Union{Nothing,KHArray3}
    # Final/seed atomic H distribution saved after the solve.
    Recomb_s::Union{Nothing,Bool}
    # Recombination flag used for the saved run.
end

KineticHInput() = KineticHInput(nothing, nothing, nothing, nothing, nothing, nothing)

mutable struct KineticHInternal
    # Internal cached arrays used by the solver. Static fields depend only on
    # mesh/velocity geometry; dynamic fields depend on current plasma/source
    # inputs and may be recomputed by run_procedure.
    vr2vx2::Union{Nothing,KHArray3}
    # vr^2 + vx^2 on the H velocity mesh at each x.
    vr2vx_vxi2::Union{Nothing,KHArray3}
    # vr^2 + (vx - vxi/vth)^2, relative to ion flow.
    fi_hat::Union{Nothing,KHArray3}
    # Normalized ion Maxwellian on the H velocity mesh.
    ErelH_P::Union{Nothing,KHArray3}
    # Relative H-proton collision energy.
    Ti_mu::Union{Nothing,KHArray3}
    # Ion temperature scaled by reduced mass factors for rate formulas.
    ni::Union{Nothing,Vector{Float64}}
    # Effective ion density profile used by the atomic solve.
    sigv::Union{Nothing,Matrix{Float64}}
    # Ionization/recombination rate table columns used by selected model.
    alpha_ion::Union{Nothing,Vector{Float64}}
    # Electron-impact ionization frequency profile.
    v_v2::Union{Nothing,KHArray5}
    # Pairwise relative velocity squared for velocity-space collision kernels.
    v_v::Union{Nothing,KHArray5}
    # Pairwise relative velocity magnitude for collision kernels.
    vr2_vx2::Union{Nothing,KHArray5}
    # Pairwise geometric velocity term used by kernel construction.
    vx_vx::Union{Nothing,Matrix{Float64}}
    # Axial velocity difference/projection term in flattened velocity space.
    Vr2pidVrdVx::Union{Nothing,Matrix{Float64}}
    # Velocity-space quadrature/Jacobian weights in flattened kernel form.
    SIG_CX::Union{Nothing,Matrix{Float64}}
    # Charge-exchange velocity-space kernel.
    SIG_H_H::Union{Nothing,Matrix{Float64}}
    # H-H elastic velocity-space kernel.
    SIG_H_H2::Union{Nothing,Matrix{Float64}}
    # H-H2 elastic velocity-space kernel.
    SIG_H_P::Union{Nothing,Matrix{Float64}}
    # H-proton elastic velocity-space kernel.
    Alpha_CX::Union{Nothing,KHArray3}
    # Charge-exchange sink frequency over (vr, vx, x).
    Alpha_H_H2::Union{Nothing,KHArray3}
    # H-H2 elastic momentum-transfer frequency over (vr, vx, x).
    Alpha_H_P::Union{Nothing,KHArray3}
    # H-proton elastic momentum-transfer frequency over (vr, vx, x).
    MH_H_sum::Union{Nothing,KHArray3}
    # Accumulated H-H elastic replacement distribution from previous iteration.
    Delta_nHs::Float64
    # Outer fixed-point density-change metric.
    Sn::Union{Nothing,KHArray3}
    # Direct atomic source distribution, including optional recombination.
    Rec::Union{Nothing,Vector{Float64}}
    # Recombination coefficient/frequency profile.
end

function KineticHInternal()::KineticHInternal
    return KineticHInternal(
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        0.0,
        nothing,
        nothing,
    )
end

mutable struct KineticHH2Moments
    # Velocity moments of the supplied molecular distribution fH2, used by
    # H-H2 elastic collision coupling.
    nH2::Union{Nothing,Vector{Float64}}
    # Molecular density profile.
    VxH2::Union{Nothing,Vector{Float64}}
    # Mean molecular axial velocity.
    TH2::Union{Nothing,Vector{Float64}}
    # Molecular temperature profile.
end

KineticHH2Moments() = KineticHH2Moments(nothing, nothing, nothing)

mutable struct KineticH
    # Main solver state for atomic hydrogen transport.
    config::KN1DConfig
    # Parsed KN1D configuration.
    collisions::KHCollisions
    # Enabled collision channels for this run.
    ion_rate_option::String
    # Ionization model selector, e.g. "janev", "collrad", or "jh".
    DeltaVx_tol::Float64
    # Floor for relative drift speeds in elastic collision frequency estimates.
    Wpp_tol::Float64
    # Floor for perpendicular/parallel energy-exchange denominator.
    CI_Test::Bool
    # Legacy consistency-test flag.
    Do_Alpha_CX_Test::Bool
    # Legacy charge-exchange alpha test flag.
    debrief_level::Int
    # Verbosity level for progress messages.
    truncate::Float64
    # Generation/outer convergence tolerance.
    max_gen::Int
    # Maximum number of collision generations.
    ni_correct::Bool
    # Whether to apply ion-density correction logic.
    compute_errors::Bool
    # Whether to compute expensive final diagnostic errors.
    recomb::Bool
    # Include recombination source when true.
    debug::Int
    # Debug verbosity/behavior flag.
    mesh::KineticMesh
    # H kinetic mesh containing x, vr, vx, Ti, Te, ne, PipeDia.
    mu::Int
    # Mass number: 1 for H, 2 for D.
    vxi::Vector{Float64}
    # Ion axial flow velocity profile.
    fHBC::Matrix{Float64}
    # Raw/possibly normalized boundary distribution on (vr, vx).
    GammaxHBC::Float64
    # Requested incoming atomic flux boundary condition.
    nvr::Int
    # Number of radial/perpendicular velocity points.
    nvx::Int
    # Number of axial velocity points.
    nx::Int
    # Number of spatial mesh points.
    vx_neg::Vector{Int}
    # Indices with vx < 0; swept right-to-left.
    vx_pos::Vector{Int}
    # Indices with vx > 0; swept left-to-right.
    vx_zero::Vector{Int}
    # Indices with vx == 0; expected empty for the transport solve.
    vth::Float64
    # Thermal velocity normalization from mesh.Tnorm and species mass.
    vr2_2vx2_2D::Matrix{Float64}
    # vr^2 - 2vx^2 helper for H-H energy-exchange diagnostics.
    dvr_vol::Vector{Float64}
    # Radial velocity quadrature weights including cylindrical volume factor.
    dvx::Vector{Float64}
    # Axial velocity quadrature weights.
    fHBC_input::Matrix{Float64}
    # Incoming positive-vx boundary distribution scaled to GammaxHBC.
    input::KineticHInput
    # Saved inputs from the latest run.
    internal::KineticHInternal
    # Cached solver internals.
    output::KineticHOutput
    # Additional diagnostics from the latest solve.
    h2_moments::KineticHH2Moments
    # Moments of the coupled molecular distribution.
    errors::KineticHErrors
    # Optional final consistency/error diagnostics.
    jh::Union{Nothing,JohnsonHinnov}
    # Johnson-Hinnov data/interpolator object when that rate model is used.
end

function KHCollisions(config::CollisionConfig)::KHCollisions
    return KHCollisions(
        config.H2_H_EL,
        config.H_H_EL,
        config.H_P_EL,
        config.H_P_CX,
        config.SIMPLE_CX,
    )
end

function KineticH(
    mesh::KineticMesh,
    mu::Integer,
    vxi::AbstractVector{<:Real},
    fHBC::AbstractMatrix{<:Real},
    GammaxHBC::Real;
    jh::Union{Nothing,JohnsonHinnov}=nothing,
    recomb::Bool=true,
    ni_correct::Bool=false,
    truncate::Real=1.0e-4,
    max_gen::Integer=100,
    compute_errors::Bool=false,
    debrief::Integer=0,
    debug::Integer=0,
    config_path::AbstractString="./config.json",
    initialize_static::Bool=false,
    )::KineticH
    mesh.mesh_type == "h" || throw(ArgumentError("KineticH requires a mesh with mesh_type == \"h\""))

    nvr = length(mesh.vr)
    nvx = length(mesh.vx)
    nx = length(mesh.x)
    length(vxi) == nx || throw(DimensionMismatch("vxi length must match mesh.x length"))
    size(fHBC) == (nvr, nvx) || throw(DimensionMismatch("fHBC must have size (length(mesh.vr), length(mesh.vx))"))

    config = get_config(String(config_path))
    collisions = KHCollisions(config.collisions)
    debrief_level = Int(debrief)
    debug_level = Int(debug)
    if debug_level > 0
        debrief_level = max(debrief_level, 1)
    end

    vxi_f = Float64.(vxi)
    fHBC_f = Float64.(fHBC)
    vx_neg = findall(<(0.0), mesh.vx)
    vx_pos = findall(>(0.0), mesh.vx)
    vx_zero = findall(==(0.0), mesh.vx)
    isempty(vx_pos) && throw(ArgumentError("mesh.vx must contain at least one positive velocity"))
    isempty(vx_neg) && throw(ArgumentError("mesh.vx must contain at least one negative velocity"))

    mu_i = Int(mu)
    vth = sqrt((2.0 * Q * mesh.Tnorm) / (mu_i * H_MASS))
    vr2_2vx2_2D = Matrix{Float64}(undef, nvr, nvx)
    @inbounds for j in 1:nvx, i in 1:nvr
        vr2_2vx2_2D[i, j] = mesh.vr[i]^2 - 2.0 * mesh.vx[j]^2
    end

    differentials = VSpaceDifferentials(mesh.vr, mesh.vx)
    kh = KineticH(
        config,
        collisions,
        config.kinetic_h.ion_rate,
        0.01,
        0.001,
        config.kinetic_h.ci_test,
        config.kinetic_h.alpha_cx_test,
        debrief_level,
        Float64(truncate),
        Int(max_gen),
        ni_correct,
        compute_errors && debrief_level > 0,
        recomb,
        debug_level,
        mesh,
        mu_i,
        vxi_f,
        fHBC_f,
        Float64(GammaxHBC),
        nvr,
        nvx,
        nx,
        vx_neg,
        vx_pos,
        vx_zero,
        vth,
        vr2_2vx2_2D,
        differentials.dvr_vol,
        differentials.dvx,
        zeros(Float64, nvr, nvx),
        KineticHInput(),
        KineticHInternal(),
        KineticHOutput(nx),
        KineticHH2Moments(),
        KineticHErrors(),
        jh,
    )

    _init_fhbc_input!(kh)
    if kh.jh === nothing && kh.ion_rate_option == "jh"
        kh.jh = JohnsonHinnov()
    end
    _test_init_parameters(kh)
    initialize_static && _init_static_internals!(kh)

    return kh
end

function _init_fhbc_input!(kh::KineticH)::Nothing
    fill!(kh.fHBC_input, 0.0)
    @inbounds for j in kh.vx_pos, i in 1:kh.nvr
        kh.fHBC_input[i, j] = kh.fHBC[i, j]
    end

    gamma_input = 1.0
    if abs(kh.GammaxHBC) > 0.0
        total = 0.0
        @inbounds for j in 1:kh.nvx
            vx_dvx = kh.mesh.vx[j] * kh.dvx[j]
            for i in 1:kh.nvr
                total += kh.dvr_vol[i] * kh.fHBC_input[i, j] * vx_dvx
            end
        end
        gamma_input = kh.vth * total
    end

    ratio = abs(kh.GammaxHBC) / gamma_input
    @inbounds for j in 1:kh.nvx, i in 1:kh.nvr
        kh.fHBC_input[i, j] *= ratio
    end
    if abs(ratio - 1.0) > 0.01 * kh.truncate
        copyto!(kh.fHBC, kh.fHBC_input)
    end
    return nothing
end

function _test_init_parameters(kh::KineticH)::Nothing
    kh.mu in (1, 2) || throw(ArgumentError("mu must be 1 for hydrogen or 2 for deuterium"))
    kh.truncate > 0.0 || throw(ArgumentError("truncate must be positive"))
    kh.max_gen > 0 || throw(ArgumentError("max_gen must be positive"))
    all(isfinite, kh.vxi) || throw(ArgumentError("vxi must contain only finite values"))
    all(isfinite, kh.fHBC) || throw(ArgumentError("fHBC must contain only finite values"))
    return nothing
end

function _as_array3_or_zeros(x, nvr::Int, nvx::Int, nx::Int, name::AbstractString)::Array{Float64,3}
    if x === nothing
        return zeros(Float64, nvr, nvx, nx)
    end
    A = Float64.(x)
    size(A) == (nvr, nvx, nx) || throw(DimensionMismatch("$name must have size ($nvr, $nvx, $nx)"))
    return Array{Float64,3}(A)
end

function _as_vector_or(x, nx::Int, default::Float64, name::AbstractString)::Vector{Float64}
    if x === nothing
        return fill(default, nx)
    end
    v = Float64.(x)
    length(v) == nx || throw(DimensionMismatch("$name length must match kh.nx"))
    return Vector{Float64}(v)
end

function run_procedure(kh::KineticH; fH2=nothing, fSH=nothing, fH=nothing, nHP=nothing, THP=nothing)::KHResults
    nvr, nvx, nx = kh.nvr, kh.nvx, kh.nx

    # Normalize optional inputs into concrete arrays so the transport kernels can
    # assume fixed shapes and Float64 storage.
    fH2_f = _as_array3_or_zeros(fH2, nvr, nvx, nx, "fH2")
    fSH_f = _as_array3_or_zeros(fSH, nvr, nvx, nx, "fSH")
    fH_f = _as_array3_or_zeros(fH, nvr, nvx, nx, "fH")
    nHP_f = _as_vector_or(nHP, nx, 0.0, "nHP")
    THP_f = _as_vector_or(THP, nx, 1.0, "THP")

    # Match Python: H2-H elastic collisions are disabled for a zero molecular distribution.
    kh.collisions = KHCollisions(
        sum(fH2_f) > 0.0 && kh.config.collisions.H2_H_EL,
        kh.collisions.H_H_EL,
        kh.collisions.H_P_EL,
        kh.collisions.H_P_CX,
        kh.collisions.SIMPLE_CX,
    )

    # Incoming atomic boundary condition: only positive-vx atoms enter from the
    # left boundary. Negative-vx atoms are determined by the backward sweep.
    @inbounds for j in kh.vx_pos, i in 1:nvr
        fH_f[i, j, 1] = kh.fHBC_input[i, j]
    end

    if kh.internal.vr2vx2 === nothing ||
       kh.internal.fi_hat === nothing ||
       kh.internal.sigv === nothing ||
       kh.internal.SIG_CX === nothing
        _init_static_internals!(kh)
    end

    # Dynamic internals depend on the current external/coupled inputs, unlike
    # static velocity geometry and collision kernels built during construction.
    _compute_dynamic_internals!(kh, fH_f, fH2_f, nHP_f, THP_f, fSH_f)

    # Initial neutral density from the supplied/boundary fH. This is the seed
    # used to form self-consistent elastic collision frequencies.
    nH = Vector{Float64}(undef, nx)
    @inbounds for k in 1:nx
        s = 0.0
        for j in 1:nvx
            dvxj = kh.dvx[j]
            for i in 1:nvr
                s += kh.dvr_vol[i] * fH_f[i, j, k] * dvxj
            end
        end
        nH[k] = s
    end

    # Side-wall loss frequency. It is local in x and proportional to radial
    # velocity divided by effective pipe diameter.
    gamma_wall = zeros(Float64, nvr, nvx, nx)
    @inbounds for k in 1:nx
        if kh.mesh.PipeDia[k] > 0.0
            inv_dia = 2.0 / kh.mesh.PipeDia[k]
            for j in 1:nvx, i in 1:nvr
                gamma_wall[i, j, k] = kh.mesh.vr[i] * inv_dia
            end
        end
    end

    # Solve the steady kinetic transport problem by directional sweeps and
    # collision/source generations.
    fH_out, nH_out, alpha_c, Beta_CX_sum, collision_freqs, m_sums =
        _run_iteration_scheme(kh, fH_f, nH, gamma_wall)

    # Convert the converged distribution into physical moments and diagnostics.
    results = _compile_results(kh, fH_out, nH_out, fSH_f, gamma_wall, alpha_c, Beta_CX_sum, collision_freqs, m_sums)

    kh.input.fH2_s = fH2_f
    kh.input.fSH_s = fSH_f
    kh.input.nHP_s = nHP_f
    kh.input.THP_s = THP_f
    kh.input.fH_s = fH_out

    _debrief_msg(kh, "Finished", 0)
    return results
end

function _unported_kinetic_h(name::Symbol)
    throw(ErrorException("KineticH.$name is scaffolded but not ported yet"))
end

function _debrief_msg(kh::KineticH, message::AbstractString, threshold::Integer)::Nothing
    debrief("Kinetic_H => " * message, kh.debrief_level > threshold)
    return nothing
end

function _init_grid!(kh::KineticH)::Nothing
    vr2vx2 = Array{Float64, 3}(undef, kh.nvr, kh.nvx, kh.nx)
    vr2vx_vxi2 = Array{Float64, 3}(undef, kh.nvr, kh.nvx, kh.nx)
    ErelH_P = Array{Float64, 3}(undef, kh.nvr, kh.nvx, kh.nx)

    coef = 0.5 * H_MASS * kh.vth^2 / Q

    @inbounds for k in 1:kh.nx
        vxi_norm = kh.vxi[k] / kh.vth
        for j in 1:kh.nvx
            vx = kh.mesh.vx[j]
            vx_shift2 = (vx - vxi_norm)^2
            vx2 = vx^2
            for i in 1:kh.nvr
                vr2 = kh.mesh.vr[i]^2
                a = vr2 + vx2
                b = vr2 + vx_shift2

                vr2vx2[i, j, k] = a
                vr2vx_vxi2[i, j, k] = b
                ErelH_P[i, j, k] = clamp(coef * b, 0.1, 2.0e4)
            end
        end
    end

    kh.internal.vr2vx2 = vr2vx2
    kh.internal.vr2vx_vxi2 = vr2vx_vxi2
    kh.internal.ErelH_P = ErelH_P

    return nothing
end

function _init_protons!(kh::KineticH)::Nothing
    _debrief_msg(kh, "Computing Ti/mu at each mesh point", 1)

    Ti_mu = Array{Float64, 3}(undef, kh.nvr, kh.nvx, kh.nx)

    @inbounds for k in 1:kh.nx
        value = kh.mesh.Ti[k] / kh.mu
        for j in 1:kh.nvx
            for i in 1:kh.nvr
                Ti_mu[i, j, k] = value
            end
        end
    end

    kh.internal.Ti_mu = Ti_mu
    kh.internal.fi_hat = create_shifted_maxwellian(kh.mesh.vr, kh.mesh.vx, kh.mesh.Ti, kh.vxi, kh.mu, 1, kh.mesh.Tnorm)
    return nothing
end

function _init_sigv!(kh::KineticH)::Nothing
    _debrief_msg(kh, "Computing sigv", 1)

    sigv = zeros(Float64, kh.nx, 3)
    
    if kh.ion_rate_option == "collrad"
        @inbounds for k in 1:kh.nx
            sigv[k, 2] = collrad_sigmav_ion_h0(kh.mesh.ne[k], kh.mesh.Te[k])
            sigv[k, 3] = sigmav_rec_h1s(kh.mesh.Te[k])
        end
    elseif kh.ion_rate_option == "jh"
        kh.jh === nothing && throw(ArgumentError("JohnsonHinnov data is required when ion_rate_option == \"jh\""))
        @inbounds for k in 1:kh.nx
            sigv[k, 2] = jhs_coef(kh.jh, kh.mesh.ne[k], kh.mesh.Te[k], no_null=true)
            sigv[k, 3] = jhalpha_coef(kh.jh, kh.mesh.ne[k], kh.mesh.Te[k], no_null=true)
        end
    elseif kh.ion_rate_option == "adas"
        @inbounds for k in 1:kh.nx
            sigv[k, 2] = scd_adas(kh.mesh.ne[k], kh.mesh.Te[k])
            sigv[k, 3] = acd_adas(kh.mesh.ne[k], kh.mesh.Te[k])
        end
    else
        @inbounds for k in 1:kh.nx
            sigv[k, 2] = sigmav_ion_h0(kh.mesh.Te[k])
            sigv[k, 3] = sigmav_rec_h1s(kh.mesh.Te[k])
        end
    end
    alpha_ion = Vector{Float64}(undef, kh.nx)
    @inbounds for k in 1:kh.nx
        alpha_ion[k] = sigv[k, 2] * kh.mesh.ne[k] / kh.vth
    end

    Rec = Vector{Float64}(undef, kh.nx)
    @inbounds for k in 1:kh.nx
        Rec[k] =  kh.mesh.ne[k] * sigv[k, 3] / kh.vth
    end

    kh.internal.sigv = sigv
    kh.internal.alpha_ion = alpha_ion
    kh.internal.Rec = Rec
    return nothing
end

function _init_v_v2!(kh::KineticH)::Nothing
    _debrief_msg(kh, "Computing compact velocity-space kernel geometry", 1)

    vx = kh.mesh.vx

    nvr = kh.nvr
    nvx = kh.nvx
    vx_diff = Array{Float64, 2}(undef, nvx, nvx)

    @inbounds for j2 in 1:nvx
        vx2 = vx[j2]
        for j1 in 1:nvx
            vx_diff[j1, j2] = vx[j1] - vx2
        end
    end

    Vr2pidVrdVx = Array{Float64, 2}(undef, nvr, nvx)

    @inbounds for j2 in 1:nvx
        dvx2 = kh.dvx[j2]
        for i2 in 1:nvr
            Vr2pidVrdVx[i2, j2] = kh.dvr_vol[i2] * dvx2
        end
    end

    kh.internal.vx_vx = vx_diff            # compact: vx_vx[j1, j2]
    kh.internal.Vr2pidVrdVx = Vr2pidVrdVx  # compact: weight[i2, j2]

    # The 5D pairwise relative-velocity arrays from the Python translation are
    # intentionally not materialized. Kernel construction below computes those
    # terms on the fly, which is faster for cold one-shot runs and saves memory.
    kh.internal.v_v2 = nothing
    kh.internal.vr2_vx2 = nothing
    kh.internal.v_v = nothing

    return nothing
end

function _init_sig_kernels!(kh::KineticH)::Nothing
    _debrief_msg(kh, "Computing fused SIG_CX, SIG_H_H, SIG_H_H2, and SIG_H_P", 1)

    vr = kh.mesh.vr
    vx = kh.mesh.vx
    nvr = kh.nvr
    nvx = kh.nvx
    ntheta = KH_NTHETA
    nvel = nvr * nvx

    scale_cx = 0.5 * H_MASS * kh.vth^2 / Q
    scale_h_h = 0.5 * H_MASS * kh.mu * kh.vth^2 / Q

    SIG_CX = Array{Float64, 2}(undef, nvel, nvel)
    SIG_H_H = Array{Float64, 2}(undef, nvel, nvel)
    SIG_H_H2 = Array{Float64, 2}(undef, nvel, nvel)
    SIG_H_P = Array{Float64, 2}(undef, nvel, nvel)

    Threads.@threads for j2 in 1:nvx
        @inbounds begin
            vx2 = vx[j2]
            for i2 in 1:nvr
                vr2 = vr[i2]
                vr2sq = vr2 * vr2
                col = i2 + (j2 - 1) * nvr
                weight = kh.internal.Vr2pidVrdVx[i2, j2]

                for j1 in 1:nvx
                    dvx = vx[j1] - vx2
                    dvx2 = dvx * dvx
                    for i1 in 1:nvr
                        vr1 = vr[i1]
                        vr1sq = vr1 * vr1
                        row = i1 + (j1 - 1) * nvr

                        s_cx = 0.0
                        s_h_h = 0.0
                        s_h_h2 = 0.0
                        s_h_p = 0.0

                        for c in 1:ntheta
                            base = vr1sq + vr2sq - 2.0 * vr1 * vr2 * KH_COS_THETA[c]
                            vv2 = base + dvx2
                            vv = sqrt(vv2)
                            dtheta = KH_DTHETA[c]

                            s_cx += vv * sigma_cx_h0(vv2 * scale_cx) * dtheta
                            s_h_h += (base - 2.0 * dvx2) * vv * sigma_el_h_h(vv2 * scale_h_h, vis=true) * dtheta / 8.0
                            s_h_h2 += vv * sigma_el_h_hh(vv2 * scale_cx) * dtheta
                            s_h_p += vv * sigma_el_p_h(vv2 * scale_cx) * dtheta
                        end

                        SIG_CX[row, col] = weight * s_cx
                        SIG_H_H[row, col] = weight * s_h_h
                        SIG_H_H2[row, col] = weight * dvx * s_h_h2
                        SIG_H_P[row, col] = weight * dvx * s_h_p
                    end
                end
            end
        end
    end

    kh.internal.SIG_CX = SIG_CX
    kh.internal.SIG_H_H = SIG_H_H
    kh.internal.SIG_H_H2 = SIG_H_H2
    kh.internal.SIG_H_P = SIG_H_P

    return nothing
end

function _init_sig_cx!(kh::KineticH)::Nothing
    _debrief_msg(kh, "Computing SIG_CX", 1)

    nvr = kh.nvr
    nvx = kh.nvx
    ncos = KH_NTHETA

    scale = 0.5 * H_MASS * kh.vth^2 / Q

    SIG_CX = Array{Float64, 2}(undef, nvr * nvx, nvr * nvx)

    @inbounds for j2 in 1:nvx
        for i2 in 1:nvr
            col = i2 + (j2 - 1) * nvr

            weight = kh.internal.Vr2pidVrdVx[i2, j2]

            for j1 in 1:nvx
                for i1 in 1:nvr
                    row = i1 + (j1 - 1) * nvr

                    s = 0.0
                    for c in 1:ncos
                        vv = kh.internal.v_v[i1, j1, i2, j2, c]
                        vv2 = kh.internal.v_v2[i1, j1, i2, j2, c]

                        s += vv * sigma_cx_h0(vv2 * scale) * KH_DTHETA[c]
                    end

                    SIG_CX[row, col] = weight * s
                end
            end
        end
    end

    kh.internal.SIG_CX = SIG_CX

    return nothing
end

function _init_sig_h_h!(kh::KineticH)::Nothing
    _debrief_msg(kh, "Computing SIG_H_H", 1)
    nvr = kh.nvr
    nvx = kh.nvx
    ntheta = KH_NTHETA
    scale = 0.5 * H_MASS * kh.mu * kh.vth^2 / Q

    SIG_H_H = Array{Float64, 2}(undef, nvr * nvx, nvr * nvx)
    @inbounds for j2 in 1:nvx
        for i2 in 1:nvr
            col = i2 + (j2 - 1) * nvr

            weight = kh.internal.Vr2pidVrdVx[i2, j2]

            for j1 in 1:nvx
                for i1 in 1:nvr
                    row = i1 + (j1 - 1) * nvr
                    s = 0.0
                    for c in 1:ntheta
                        vv = kh.internal.v_v[i1, j1, i2, j2, c]
                        vv2 = kh.internal.v_v2[i1, j1, i2, j2, c]
                        vr2vx2 = kh.internal.vr2_vx2[i1, j1, i2, j2, c]
                        s += vr2vx2 * vv * sigma_el_h_h(vv2 * scale, vis=true) * KH_DTHETA[c] / 8.0
                    end

                    SIG_H_H[row, col] = weight * s
                end
            end
        end
    end

    kh.internal.SIG_H_H = SIG_H_H
    return nothing
end

function _init_sig_h_h2!(kh::KineticH)::Nothing
    _debrief_msg(kh, "Computing SIG_H_H2", 1)
    nvr = kh.nvr
    nvx = kh.nvx
    ntheta = KH_NTHETA
    scale = 0.5 * H_MASS * kh.vth^2 / Q

    SIG_H_H2 = Array{Float64, 2}(undef, nvr * nvx, nvr * nvx)
    @inbounds for j2 in 1:nvx
        for i2 in 1:nvr
            col = i2 + (j2 - 1) * nvr
            weight = kh.internal.Vr2pidVrdVx[i2, j2]
            for j1 in 1:nvx
                vx_vx = kh.internal.vx_vx[j1, j2]
                for i1 in 1:nvr
                    row = i1 + (j1 - 1) * nvr
                    s = 0.0
                    for c in 1:ntheta
                        vv = kh.internal.v_v[i1, j1, i2, j2, c]
                        vv2 = kh.internal.v_v2[i1, j1, i2, j2, c]
                        s += vv * sigma_el_h_hh(vv2 * scale) * KH_DTHETA[c]
                    end

                    SIG_H_H2[row, col] = weight * vx_vx * s
                end
            end
        end
    end

    kh.internal.SIG_H_H2 = SIG_H_H2
    return nothing
end

function _init_sig_h_p!(kh::KineticH)::Nothing
    _debrief_msg(kh, "Computing SIG_H_P", 1)
    nvr = kh.nvr
    nvx = kh.nvx
    ntheta = KH_NTHETA
    scale = 0.5 * H_MASS * kh.vth^2 / Q

    SIG_H_P = Array{Float64, 2}(undef, nvr * nvx, nvr * nvx)
    @inbounds for j2 in 1:nvx
        for i2 in 1:nvr
            col = i2 + (j2 - 1) * nvr

            weight = kh.internal.Vr2pidVrdVx[i2, j2]

            for j1 in 1:nvx
                vx_vx = kh.internal.vx_vx[j1, j2]
                for i1 in 1:nvr
                    row = i1 + (j1 - 1) * nvr
                    s = 0.0
                    for c in 1:ntheta
                        vv = kh.internal.v_v[i1, j1, i2, j2, c]
                        vv2 = kh.internal.v_v2[i1, j1, i2, j2, c]
                        s += vv * sigma_el_p_h(vv2 * scale) * KH_DTHETA[c]
                    end

                    SIG_H_P[row, col] = weight * vx_vx * s
                end
            end
        end
    end

    kh.internal.SIG_H_P = SIG_H_P
    return nothing
end

function _init_static_internals!(kh::KineticH)::Nothing
    _init_grid!(kh)
    _init_protons!(kh)
    _init_sigv!(kh)
    _init_v_v2!(kh)
    _init_sig_kernels!(kh)
    return nothing
end

function _compute_ni!(kh::KineticH, nHP::Vector{Float64})::Nothing
    _debrief_msg(kh, "Computing ni from nHP", 1)

    length(nHP) == kh.nx || throw(DimensionMismatch("nHP length must match kh.nx"))
    ni = Vector{Float64}(undef, kh.nx)
    if kh.ni_correct
        @inbounds for k in 1:kh.nx
            ni[k] = kh.mesh.ne[k] - Float64(nHP[k])
        end
    else
        copyto!(ni, kh.mesh.ne)
    end
    @inbounds for k in 1:kh.nx
        floor_ne = 0.01 * kh.mesh.ne[k]
        if ni[k] < floor_ne
            ni[k] = floor_ne
        end
    end
    kh.internal.ni = ni
    return nothing
end

function _compute_sn!(kh::KineticH, fSH::Array{Float64,3})::Nothing
    _debrief_msg(kh, "Computing Sn", 1)
    nvr = kh.nvr
    nvx = kh.nvx
    nx = kh.nx
    size(fSH) == (nvr, nvx, nx) || throw(DimensionMismatch("fSH must have size (kh.nvr, kh.nvx, kh.nx)"))
    kh.recomb && kh.internal.fi_hat === nothing && throw(ArgumentError("kh.internal.fi_hat must be initialized before _compute_sn!"))
    kh.recomb && kh.internal.ni === nothing && throw(ArgumentError("kh.internal.ni must be initialized before _compute_sn!"))
    kh.recomb && kh.internal.Rec === nothing && throw(ArgumentError("kh.internal.Rec must be initialized before _compute_sn!"))

    Sn = Array{Float64, 3}(undef, nvr, nvx, nx)
    @inbounds for k in 1:nx
        recomb_weight = kh.recomb ? kh.internal.ni[k] * kh.internal.Rec[k] : 0.0
        for j in 1:nvx
            for i in 1:nvr
                value = fSH[i, j, k] / kh.vth
                if kh.recomb
                    value += kh.internal.fi_hat[i, j, k] * recomb_weight
                end
                Sn[i, j, k] = value
            end
        end
    end
    kh.internal.Sn = Sn
    return nothing
end

function _compute_fh2_moments!(kh::KineticH, fH2::Array{Float64, 3})::Nothing
    _debrief_msg(kh, "Computing vx and T moments of fH2", 1)
    size(fH2) == (kh.nvr, kh.nvx, kh.nx) || throw(DimensionMismatch("fH2 must have size (kh.nvr, kh.nvx, kh.nx)"))

    nvr = kh.nvr
    nvx = kh.nvx
    nx = kh.nx

    vr = kh.mesh.vr
    vx = kh.mesh.vx

    nH2 = zeros(Float64, nx)
    VxH2 = zeros(Float64, nx)
    TH2 = fill(1.0, nx)

    @inbounds for k in 1:nx
        n_sum = 0.0
        vx_sum = 0.0

        for j in 1:nvx
            vxj = vx[j]
            dvxj = kh.dvx[j]

            for i in 1:nvr
                w = kh.dvr_vol[i] * dvxj
                f = fH2[i, j, k]

                n_sum += f * w
                vx_sum += f * vxj * w
            end
        end

        nH2[k] = n_sum

        if n_sum <= 0.0
            continue
        end

        VxH2_k = kh.vth * vx_sum / n_sum
        VxH2[k] = VxH2_k

        vx_shift = VxH2_k / kh.vth
        temp_sum = 0.0

        for j in 1:nvx
            vxran = vx[j] - vx_shift
            vxran2 = vxran * vxran
            dvxj = kh.dvx[j]

            for i in 1:nvr
                vri = vr[i]
                vr2vx2_ran2 = vri * vri + vxran2
                w = kh.dvr_vol[i] * dvxj

                temp_sum += vr2vx2_ran2 * fH2[i, j, k] * w
            end
        end

        TH2[k] =
            (2.0 * kh.mu * H_MASS) *
            kh.vth^2 *
            temp_sum /
            (3.0 * Q * n_sum)
    end

    kh.h2_moments.nH2 = nH2
    kh.h2_moments.VxH2 = VxH2
    kh.h2_moments.TH2 = TH2

    return nothing
end

function _compute_alpha_cx!(kh::KineticH)::Nothing
    _debrief_msg(kh, "Computing Alpha_CX", 1)

    nvr = kh.nvr
    nvx = kh.nvx
    nx = kh.nx

    Alpha_CX = Array{Float64, 3}(undef, nvr, nvx, nx)
    kh.internal.ni === nothing && throw(ArgumentError("kh.internal.ni must be initialized before _compute_alpha_cx!"))

    if kh.collisions.SIMPLE_CX
        @inbounds for k in 1:nx
            ni_k = kh.internal.ni[k]
            for j in 1:nvx
                for i in 1:nvr
                    Alpha_CX[i, j, k] =
                        sigmav_cx_h0(
                            kh.internal.Ti_mu[i, j, k],
                            kh.internal.ErelH_P[i, j, k],
                        ) / kh.vth * ni_k
                end
            end
        end
    else
        work = Vector{Float64}(undef, nvr * nvx)
        _mul_scaled_kernel_slices!(Alpha_CX, kh.internal.SIG_CX, kh.internal.fi_hat, kh.internal.ni, work)
    end

    kh.internal.Alpha_CX = Alpha_CX

    return nothing
end

function _mul_kernel_slices!(
    out::Array{Float64,3},
    kernel::Matrix{Float64},
    src::Array{Float64,3},
    work::Vector{Float64},
)::Nothing
    nvr, nvx, nx = size(out)
    nvel = nvr * nvx
    length(work) == nvel || throw(DimensionMismatch("work length must equal nvr * nvx"))

    @inbounds for k in 1:nx
        src_k = @view src[:, :, k]
        for idx in 1:nvel
            work[idx] = src_k[idx]
        end

        out_k = reshape(@view(out[:, :, k]), nvel)
        mul!(out_k, kernel, work)
    end

    return nothing
end

function _mul_scaled_kernel_slices!(
    out::Array{Float64,3},
    kernel::Matrix{Float64},
    src::Array{Float64,3},
    scale::Vector{Float64},
    work::Vector{Float64},
)::Nothing
    nvr, nvx, nx = size(out)
    nvel = nvr * nvx
    length(work) == nvel || throw(DimensionMismatch("work length must equal nvr * nvx"))
    length(scale) == nx || throw(DimensionMismatch("scale length must match nx"))

    @inbounds for k in 1:nx
        src_k = @view src[:, :, k]
        scale_k = scale[k]
        for idx in 1:nvel
            work[idx] = src_k[idx] * scale_k
        end

        out_k = reshape(@view(out[:, :, k]), nvel)
        mul!(out_k, kernel, work)
    end

    return nothing
end

function _compute_alpha_h_h2!(kh::KineticH, fH2::Array{Float64, 3})::Nothing
    _debrief_msg(kh, "Computing Alpha_H_H2", 1)
    size(fH2) == (kh.nvr, kh.nvx, kh.nx) || throw(DimensionMismatch("fH2 must have size (kh.nvr, kh.nvx, kh.nx)"))

    nvr = kh.nvr
    nvx = kh.nvx
    nx = kh.nx

    Alpha_H_H2 = Array{Float64, 3}(undef, nvr, nvx, nx)
    work = Vector{Float64}(undef, nvr * nvx)
    _mul_kernel_slices!(Alpha_H_H2, kh.internal.SIG_H_H2, fH2, work)

    kh.internal.Alpha_H_H2 = Alpha_H_H2
    return nothing
end

function _compute_alpha_h_p!(kh::KineticH)::Nothing
    _debrief_msg(kh, "Computing Alpha_H_P", 1)

    nvr = kh.nvr
    nvx = kh.nvx
    nx = kh.nx

    Alpha_H_P = Array{Float64, 3}(undef, nvr, nvx, nx)
    work = Vector{Float64}(undef, nvr * nvx)
    _mul_scaled_kernel_slices!(Alpha_H_P, kh.internal.SIG_H_P, kh.internal.fi_hat, kh.internal.ni, work)

    kh.internal.Alpha_H_P = Alpha_H_P
    return nothing
end

function _compute_dynamic_internals!(
    kh::KineticH,
    fH::Array{Float64,3},
    fH2::Array{Float64,3},
    nHP::Vector{Float64},
    THP::Vector{Float64},
    fSH::Array{Float64,3},
)::Nothing
    New_Molecular_Ions = true

    if (kh.input.nHP_s !== nothing) && (kh.input.nHP_s == nHP) && (kh.input.THP_s == THP)
        New_Molecular_Ions = false
    end
    New_fH2 = true
    if (kh.input.fH2_s !== nothing) && (kh.input.fH2_s == fH2)
        New_fH2 = false
    end
    New_H_Seed = true
    if (kh.input.fH_s !== nothing) && (kh.input.fH_s == fH)
        New_H_Seed = false
    end

    kh.h2_moments.nH2 = zeros(Float64, kh.nx)
    kh.h2_moments.VxH2 = zeros(Float64, kh.nx)
    kh.h2_moments.TH2 = fill(1.0, kh.nx)
    
    if New_H_Seed
        kh.internal.MH_H_sum = zeros(Float64, kh.nvr, kh.nvx, kh.nx)
        kh.internal.Delta_nHs = 1.0
    end

    fH2_sum = 0.0
    @inbounds for i in eachindex(fH2)
        fH2_sum += fH2[i]
    end
    if New_fH2 && (fH2_sum > 0.0)
        _compute_fh2_moments!(kh, fH2)
    end
    if New_Molecular_Ions
        _compute_ni!(kh, nHP)
    end
    _compute_sn!(kh, fSH)
    if ((kh.internal.Alpha_CX === nothing) || New_Molecular_Ions) && (kh.collisions.H_P_CX)
        _compute_alpha_cx!(kh)
    end
    if ((kh.internal.Alpha_H_H2 === nothing) || New_fH2) && kh.collisions.H2_H_EL
        _compute_alpha_h_h2!(kh, fH2)
    end
    if ((kh.internal.Alpha_H_P === nothing) || New_Molecular_Ions) && kh.collisions.H_P_EL
        _compute_alpha_h_p!(kh)
    end
    return nothing
end
function _compute_omega_values(kh::KineticH, fH::Array{Float64,3}, nH::Vector{Float64})::CollisionType{Vector{Float64}}
    nvr, nvx, nx = kh.nvr, kh.nvx, kh.nx

    # Effective elastic collision frequencies for the current neutral state.
    # These are x-only scalars produced by velocity-space moment balances.
    Omega_H_P = zeros(Float64, nx)
    Omega_H_H2 = zeros(Float64, nx)
    Omega_H_H = zeros(Float64, nx)

    vth = kh.vth

    if any(nH .<= 0.0)
        return CollisionType(Omega_H_H, Omega_H_P, Omega_H_H2)
    end
    VxH = Vector{Float64}(undef, nx)
    if kh.collisions.H_P_EL || kh.collisions.H2_H_EL || kh.collisions.H_H_EL
        # Mean neutral flow needed to compare neutral motion with proton/H2
        # background flow in the elastic collision models.
        @inbounds for k in 1:nx
            vx_sum = 0.0
            for j in 1:nvx 
                vxj = kh.mesh.vx[j]
                dvxj = kh.dvx[j]
                for i in 1:nvr
                    f = fH[i, j, k]
                    w = kh.dvr_vol[i] * dvxj * vxj
                    vx_sum += f * w
                end
            end
            VxH[k] = vth * vx_sum / nH[k]
        end
    end
    if kh.collisions.H_P_EL
        _debrief_msg(kh, "Computing Omega_H_P", 1)

        @inbounds for k in 1:nx
            # Avoid division by a near-zero drift speed while preserving sign,
            # matching the Python/IDL tolerance behavior.
            raw = (VxH[k] - kh.vxi[k]) / kh.vth
            mag = max(abs(raw), kh.DeltaVx_tol)
            delta = raw >= 0.0 ? mag : -mag

            s = 0.0
            for j in 1:nvx
                dvxj = kh.dvx[j]
                for i in 1:nvr
                    s += kh.dvr_vol[i] *
                        kh.internal.Alpha_H_P[i, j, k] *
                        fH[i, j, k] *
                        dvxj
                end
            end

            Omega_H_P[k] = max(s / (nH[k] * delta), 0.0)
        end
    end

    if kh.collisions.H2_H_EL
        _debrief_msg(kh, "Computing Omega_H_H2", 1)

        @inbounds for k in 1:nx
            raw = (VxH[k] - kh.h2_moments.VxH2[k]) / kh.vth
            mag = max(abs(raw), kh.DeltaVx_tol)
            delta = raw >= 0.0 ? mag : -mag

            s = 0.0
            for j in 1:nvx
                dvxj = kh.dvx[j]
                for i in 1:nvr
                    s += kh.dvr_vol[i] *
                        kh.internal.Alpha_H_H2[i, j, k] *
                        fH[i, j, k] *
                        dvxj
                end
            end

            Omega_H_H2[k] = max(s / (nH[k] * delta), 0.0)
        end
    end        

    if kh.collisions.H_H_EL
        _debrief_msg(kh, "Computing Omega_H_H", 1)

        # H-H elastic scattering uses a perpendicular/parallel energy exchange
        # moment. On later outer iterations, the accumulated MH_H source from
        # the previous solve is used to form this balance.
        Wperp_paraH = zeros(Float64, nx)

        if sum(kh.internal.MH_H_sum) <= 0.0
            @inbounds for k in 1:nx
                s = 0.0
                VxH_k = VxH[k]

                for j in 1:nvx
                    vxran = kh.mesh.vx[j] - VxH_k
                    vxterm = 2.0 * vxran * vxran
                    dvxj = kh.dvx[j]

                    for i in 1:nvr
                        vri = kh.mesh.vr[i]
                        vr2_2vx_ran2 = vri * vri - vxterm

                        s += kh.dvr_vol[i] *
                            vr2_2vx_ran2 *
                            fH[i, j, k] *
                            dvxj
                    end
                end

                Wperp_paraH[k] = s / nH[k]
            end
        else
            @inbounds for k in 1:nx
                s = 0.0

                for j in 1:nvx
                    dvxj = kh.dvx[j]

                    for i in 1:nvr
                        M_fH = kh.internal.MH_H_sum[i, j, k] - fH[i, j, k]

                        s += kh.dvr_vol[i] *
                            kh.vr2_2vx2_2D[i, j] *
                            M_fH *
                            dvxj
                    end
                end

                Wperp_paraH[k] = -s / nH[k]
            end
        end
        @inbounds for k in 1:nx
            s = 0.0

            for j in 1:nvx
                dvxj = kh.dvx[j]

                for i in 1:nvr
                    row = i + (j - 1) * nvr

                    # Apply the flattened velocity-space collision kernel using
                    # Julia's column-major order, equivalent to NumPy
                    # reshape(..., order="F") in the reference implementation.
                    alpha_ij = 0.0
                    for j2 in 1:nvx
                        for i2 in 1:nvr
                            col = i2 + (j2 - 1) * nvr
                            alpha_ij += kh.internal.SIG_H_H[row, col] * fH[i2, j2, k]
                        end
                    end

                    s += kh.dvr_vol[i] *
                         alpha_ij *
                         fH[i, j, k] *
                         dvxj
                end
            end

            raw = Wperp_paraH[k]
            mag = max(abs(raw), kh.Wpp_tol)
            Wpp = raw >= 0.0 ? mag : -mag

            Omega_H_H[k] = max(s / (nH[k] * Wpp), 0.0)
        end
    end
    return CollisionType(Omega_H_H, Omega_H_P, Omega_H_H2)
end
      
function _compute_collision_frequency(kh::KineticH, collision_freqs::CollisionType{Vector{Float64}}, gamma_wall::Array{Float64, 3})::Array{Float64, 3}
    nvr = kh.nvr
    nvx = kh.nvx
    nx = kh.nx

    # Total loss frequency in the transport equation: ionization, wall loss,
    # charge-exchange sink when enabled, and elastic scattering loss.
    Omega_EL = collision_freqs.H_P + collision_freqs.H_H2 + collision_freqs.H_H
    alpha_c = Array{Float64, 3}(undef, nvr, nvx, nx)
    if kh.collisions.H_P_CX
        @inbounds for k in 1:nx
            for j in 1:nvx
                for i in 1:nvr
                    alpha_c[i, j, k] = kh.internal.Alpha_CX[i, j, k] + kh.internal.alpha_ion[k] + Omega_EL[k] + gamma_wall[i, j, k]
                end
            end
        end
    else
        @inbounds for k in 1:nx
            for j in 1:nvx
                for i in 1:nvr
                    alpha_c[i, j, k] = kh.internal.alpha_ion[k] + Omega_EL[k] + gamma_wall[i, j, k]
                end
            end
        end
    end

    _test_grid_spacing(kh, alpha_c)
    return alpha_c
end

function _compute_collisions_frequency(kh::KineticH, collision_freqs::CollisionType{Vector{Float64}}, gamma_wall::Array{Float64, 3})::Array{Float64, 3}
    return _compute_collision_frequency(kh, collision_freqs, gamma_wall)
end

function _test_grid_spacing(kh::KineticH, alpha_c::Array{Float64,3})::Nothing
    _debrief_msg(kh, "Testing x grid spacing", 1)

    # The sweep formula assumes the spatial grid is fine enough relative to the
    # local loss frequency. Store the limiting dx for diagnostics/errors.
    max_dx_full = fill(1.0e32, kh.nx)
    @inbounds for k in 1:kh.nx
        for j in kh.vx_pos
            vx2 = 2.0 * kh.mesh.vx[j]
            for i in 1:kh.nvr
                local_dx = vx2 / alpha_c[i, j, k]
                if isfinite(local_dx) && local_dx > 0.0 && local_dx < max_dx_full[k]
                    max_dx_full[k] = local_dx
                end
            end
        end
    end

    max_dx = Vector{Float64}(undef, max(kh.nx - 1, 0))
    @inbounds for k in 1:length(max_dx)
        max_dx[k] = min(max_dx_full[k], max_dx_full[k + 1])
    end
    kh.errors.Max_dx = max_dx

    @inbounds for k in 1:length(max_dx)
        dx = kh.mesh.x[k + 1] - kh.mesh.x[k]
        if max_dx[k] < dx
            throw(ErrorException("Kinetic_H => x mesh spacing is too large at interval $k"))
        end
    end
    return nothing
end

function _compute_mesh_equation_coefficients(
    kh::KineticH,
    alpha_c::Array{Float64, 3},
)::MeshEqCoefficients

    nvr = kh.nvr
    nvx = kh.nvx
    nx = kh.nx

    Ak = zeros(Float64, nvr, nvx, nx)
    Bk = zeros(Float64, nvr, nvx, nx)
    Ck = zeros(Float64, nvr, nvx, nx)
    Dk = zeros(Float64, nvr, nvx, nx)
    Fk = zeros(Float64, nvr, nvx, nx)
    Gk = zeros(Float64, nvr, nvx, nx)

    # Discretized steady transport coefficients. Positive-vx particles march
    # left-to-right with A/B/F; negative-vx particles march right-to-left with
    # C/D/G. F and G carry the direct atomic source Sn.
    @inbounds for k in 1:(nx - 1)
        xdiff = kh.mesh.x[k + 1] - kh.mesh.x[k]

        for j in kh.vx_pos
            vxj = kh.mesh.vx[j]

            for i in 1:nvr
                denom = 2.0 * vxj + xdiff * alpha_c[i, j, k + 1]

                Ak[i, j, k] =
                    (2.0 * vxj - xdiff * alpha_c[i, j, k]) / denom

                Bk[i, j, k] = xdiff / denom

                Fk[i, j, k] =
                    xdiff *
                    (kh.internal.Sn[i, j, k + 1] + kh.internal.Sn[i, j, k]) /
                    denom
            end
        end
        for j in kh.vx_neg
            vxj = kh.mesh.vx[j]

            for i in 1:nvr
                denom = -2.0 * vxj + xdiff * alpha_c[i, j, k]

                Ck[i, j, k + 1] =
                    (-2.0 * vxj - xdiff * alpha_c[i, j, k + 1]) / denom

                Dk[i, j, k + 1] = xdiff / denom

                Gk[i, j, k + 1] =
                    xdiff *
                    (kh.internal.Sn[i, j, k + 1] + kh.internal.Sn[i, j, k]) /
                    denom
            end
        end
    end
    return MeshEqCoefficients(Ak, Bk, Ck, Dk, Fk, Gk)
end

function _compute_beta_cx(kh::KineticH, fH::Array{Float64, 3})::Array{Float64, 3}

    nvr = kh.nvr
    nvx = kh.nvx
    nx = kh.nx

    Beta_CX = Array{Float64, 3}(undef, nvr, nvx, nx)
    if kh.collisions.H_P_CX
        _debrief_msg(kh, "Computing Beta_CX", 1)
        if kh.collisions.SIMPLE_CX
            # Simple CX model: integrate the previous-generation CX loss rate,
            # then redistribute newly born neutrals with the ion Maxwellian
            # shape fi_hat at the same x.
            @inbounds for k in 1:nx
                source_sum = 0.0
                for j in 1:nvx
                    dvxj = kh.dvx[j]
                    for i in 1:nvr
                        source_sum += kh.dvr_vol[i] *
                                      kh.internal.Alpha_CX[i, j, k] *
                                      fH[i, j, k] *
                                      dvxj
                    end
                end
                for j in 1:nvx
                    for i in 1:nvr
                        Beta_CX[i, j, k] = kh.internal.fi_hat[i, j, k] * source_sum
                    end
                end
            end
        else
            # Full CX model: apply the velocity-space cross-section kernel to
            # the previous generation before weighting by the local ion density.
            @inbounds for k in 1:nx
                ni_k = kh.internal.ni[k]

                for j in 1:nvx
                    for i in 1:nvr
                        row = i + (j - 1) * nvr
                        s = 0.0

                        for j2 in 1:nvx
                            for i2 in 1:nvr
                                col = i2 + (j2 - 1) * nvr
                                s += kh.internal.SIG_CX[row, col] * fH[i2, j2, k]
                            end
                        end

                        Beta_CX[i, j, k] =
                            ni_k * kh.internal.fi_hat[i, j, k] * s
                    end
                end
            end
        end
    end

    return Beta_CX
end

function _compute_mh_values(kh::KineticH, fH::Array{Float64, 3}, nH::AbstractVector{Float64})::CollisionType{Array{Float64,3}}
    nvr = kh.nvr
    nvx = kh.nvx
    nx = kh.nx
    vth = kh.vth
    MH_H = zeros(Float64, nvr, nvx, nx)
    MH_P = zeros(Float64, nvr, nvx, nx)
    MH_H2 = zeros(Float64, nvr, nvx, nx)
    VxHG = zeros(Float64, nx)
    THG = zeros(Float64, nx)
    
    if kh.collisions.H_H_EL || kh.collisions.H_P_EL || kh.collisions.H2_H_EL
        # Moment-match the current generation: elastic collision sources are
        # modeled as shifted Maxwellians whose density, flow, and temperature
        # are derived from this generation.
        @inbounds for k in 1:nx
            vx_sum = 0.0
            nHk = nH[k]
            for j in 1:nvx 
                vxj = kh.mesh.vx[j]
                dvxj = kh.dvx[j]
                for i in 1:nvr
                    f = fH[i, j, k]
                    vx_sum += f * kh.dvr_vol[i] * dvxj * vxj
                end
            end
            VxHG[k] = vth * vx_sum / nHk

            temp_sum = 0.0
            vx_shift = VxHG[k] / vth
            for j in 1:nvx
                vxran2 = (kh.mesh.vx[j] - vx_shift)^2
                dvxj = kh.dvx[j]
                for i in 1:nvr
                    vr2vx2_ran2 = kh.mesh.vr[i]^2 + vxran2
                    temp_sum += kh.dvr_vol[i] * vr2vx2_ran2 * fH[i, j, k] * dvxj
                end
            end
            THG[k] = kh.mu * H_MASS * vth^2 * temp_sum / (3.0 * Q * nHk)
        end
        if kh.collisions.H_H_EL
            _debrief_msg(kh, "Computing MH_H", 1)

            # H-H scattering relaxes toward a Maxwellian with the generation's
            # own flow and temperature.
            Maxwell = create_shifted_maxwellian(kh.mesh.vr, kh.mesh.vx, THG, VxHG, kh.mu, 1, kh.mesh.Tnorm)
            @inbounds for k in 1:nx
                for j in 1:nvx
                    for i in 1:nvr
                        MH_H[i, j, k] = Maxwell[i, j, k] * nH[k]
                    end
                end
            end
        end
        if kh.collisions.H_P_EL
            _debrief_msg(kh, "Computing MH_P", 1)

            # H-P scattering relaxes toward a mixture of neutral-generation and
            # ion background moments.
            vx_shift = Vector{Float64}(undef, nx)
            Tmaxwell = Vector{Float64}(undef, nx)
            for k in 1:nx
                vx_shift[k] = (VxHG[k] + kh.vxi[k]) / 2.0
                Tmaxwell[k] = THG[k] + (2.0/4.0) * (kh.mesh.Ti[k] - THG[k] + kh.mu * H_MASS * ((kh.vxi[k] - VxHG[k])^2) / (6*Q))
            end
            Maxwell = create_shifted_maxwellian(kh.mesh.vr, kh.mesh.vx, Tmaxwell, vx_shift, kh.mu, 1, kh.mesh.Tnorm)
            @inbounds for k in 1:nx
                for j in 1:nvx
                    for i in 1:nvr
                        MH_P[i, j, k] = Maxwell[i, j, k] * nH[k]
                    end
                end
            end
        end
        if kh.collisions.H2_H_EL
            _debrief_msg(kh, "Computing MH_H2", 1)

            # H-H2 scattering uses molecular flow/temperature moments from the
            # coupled molecular distribution.
            vx_shift = Vector{Float64}(undef, nx)
            Tmaxwell = Vector{Float64}(undef, nx)
            for k in 1:nx
                vx_shift[k] = (VxHG[k] + 2.0 * kh.h2_moments.VxH2[k]) / 3.0
                Tmaxwell[k] = THG[k] + (4.0/9.0) * (kh.h2_moments.TH2[k] - THG[k] + 2*kh.mu * H_MASS * ((kh.h2_moments.VxH2[k] - VxHG[k])^2) / (6*Q))
            end
            Maxwell = create_shifted_maxwellian(kh.mesh.vr, kh.mesh.vx, Tmaxwell, vx_shift, kh.mu, 1, kh.mesh.Tnorm)
            @inbounds for k in 1:nx
                for j in 1:nvx
                    for i in 1:nvr
                        MH_H2[i, j, k] = Maxwell[i, j, k] * nH[k]
                    end
                end
            end
        end
    end
    return CollisionType(MH_H, MH_P, MH_H2)
end

function _run_generations(
    kh::KineticH,
    fH::Array{Float64,3},
    nH::Vector{Float64},
    fHG::Array{Float64,3},
    NHG::Array{Float64,2},
    meq_coeffs::MeshEqCoefficients,
    collision_freqs::CollisionType{Vector{Float64}},
    fH_iterate::Bool)::Tuple{Array{Float64,3},
    Vector{Float64},
    Array{Float64,3},
    Array{Float64,2},
    Array{Float64,3},
    CollisionType,
    Int
}
    nvr = kh.nvr
    nvx = kh.nvx
    nx = kh.nx

    Beta_CX_sum = zeros(Float64, nvr, nvx, nx)

    m_sums = CollisionType(
        zeros(Float64, nvr, nvx, nx),
        zeros(Float64, nvr, nvx, nx),
        zeros(Float64, nvr, nvx, nx),
    )

    fH_generations = fH_iterate || kh.collisions.H_P_CX

    # Inner generation loop. Each pass turns the current generation fHG into
    # charge-exchange and elastic-collision sources, sweeps those sources through
    # the mesh, and adds the resulting next generation to the total fH.
    igen = 0
    while true
        if igen >= kh.max_gen
            error(
                "Kinetic_H: failed to converge after $(kh.max_gen) generations. " *
                "The $(kh.max_gen)th generation is still contributing a non-negligible amount " *
                "to the total neutral density. This means there are neutrals undergoing " *
                "$(kh.max_gen) charge exchange or scattering events before ionisation, which " *
                "is unlikely in typical tokamak conditions and probably indicates a problem " *
                "with the input profiles."
            )
        end
        if !fH_generations
            break
        end
        igen += 1
        _debrief_msg(kh, "Computing atomic neutral generation#$(igen)", 0)

        # Charge exchange creates a source of new neutrals from the previous
        # generation. Accumulate it for final source/momentum diagnostics.
        Beta_CX = _compute_beta_cx(kh, fHG)
        @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr 
            Beta_CX_sum[i, j, k] += Beta_CX[i, j, k]
        end

        # Elastic collisions are represented as source distributions MH_* built
        # from the previous generation's moments.
        m_vals = _compute_mh_values(kh, fHG, view(NHG, :, igen))

        @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
            m_sums.H_H[i, j, k] += m_vals.H_H[i, j, k]
            m_sums.H_P[i, j, k] += m_vals.H_P[i, j, k]
            m_sums.H_H2[i, j, k] += m_vals.H_H2[i, j, k]
        end

        # Convert elastic replacement distributions into source strength using
        # the effective elastic collision frequencies computed by the outer loop.
        OmegaM = Array{Float64,3}(undef, nvr, nvx, nx)
        @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
            OmegaM[i, j, k] = collision_freqs.H_H[k] * m_vals.H_H[i, j, k] +
                collision_freqs.H_P[k] * m_vals.H_P[i, j, k] +
                collision_freqs.H_H2[k] * m_vals.H_H2[i, j, k]
        end

        # Build the next generation by transporting the new source terms.
        fill!(fHG, 0.0)
        @inbounds for k in 1:(nx-1)
            kp1 = k + 1
            for j in kh.vx_pos
                for i in 1:nvr
                    fHG[i, j, kp1] =
                        meq_coeffs.A[i, j, k] * fHG[i, j, k] +
                        meq_coeffs.B[i, j, k] *
                        (
                            Beta_CX[i, j, kp1] + OmegaM[i, j, kp1] +
                            Beta_CX[i, j, k]   + OmegaM[i, j, k]
                        )
                end
            end
        end

        # Backward sweep for negative vx.
        @inbounds for k in nx:-1:2
            km1 = k - 1

            for j in kh.vx_neg
                for i in 1:nvr
                    fHG[i, j, km1] =
                        meq_coeffs.C[i, j, k] * fHG[i, j, k] +
                        meq_coeffs.D[i, j, k] *
                        (
                            Beta_CX[i, j, km1] + OmegaM[i, j, km1] +
                            Beta_CX[i, j, k]   + OmegaM[i, j, k]
                        )
                end
            end
        end

        @inbounds for k in 1:nx
            s = 0.0
            for j in 1:nvx
                dvxj = kh.dvx[j]
                for i in 1:nvr
                    s += kh.dvr_vol[i] * fHG[i, j, k] * dvxj
                end
            end
            NHG[k, igen + 1] = s
        end

        # Add this generation to the total neutral distribution and density.
        @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
            fH[i, j, k] += fHG[i, j, k]
        end

        @inbounds for k in 1:nx
            nH[k] += NHG[k, igen + 1]
        end

        max_nH = maximum(nH)
        Delta_nHG = maximum(@view NHG[:, igen + 1]) / max_nH

        # Stop when the newest generation is small relative to the total density.
        # During outer fixed-point iteration, do not over-solve the inner series
        # far beyond the current outer density error.
        if (Delta_nHG < kh.truncate) ||
           (fH_iterate && Delta_nHG < 0.003 * kh.internal.Delta_nHs)
            break
        end
    end

    return fH, nH, fHG, NHG, Beta_CX_sum, m_sums, igen
end

function _run_iteration_scheme(
    kh::KineticH,
    fH::Array{Float64,3},
    nH::Vector{Float64},
    gamma_wall::Array{Float64, 3}
    )::Tuple{Array{Float64,3}, 
    Vector{Float64}, 
    Array{Float64,3}, 
    Array{Float64, 3},
    CollisionType{Vector{Float64}},
    CollisionType{Array{Float64,3}}
    }
    nvr = kh.nvr
    nvx = kh.nvx
    nx = kh.nx

    fH_iterate = false
    if kh.collisions.H_H_EL || kh.collisions.H_P_EL || kh.collisions.H2_H_EL
        fH_iterate = true
    end

    fHG = zeros(Float64, nvr, nvx, nx)
    NHG = zeros(Float64, nx, kh.max_gen + 1)
    alpha_c = zeros(Float64, nvr, nvx, nx)
    Beta_CX_sum = zeros(Float64, nvr, nvx, nx)
    collision_freqs = CollisionType(zeros(Float64, nx), zeros(Float64, nx), zeros(Float64, nx))
    m_sums = CollisionType(
        zeros(Float64, nvr, nvx, nx),
        zeros(Float64, nvr, nvx, nx),
        zeros(Float64, nvr, nvx, nx),
    )
    igen = 0

    # Outer fixed-point loop. Elastic collision frequencies depend on the
    # current neutral distribution, so recompute the transport operator from fH
    # and repeat until the density profile is self-consistent.
    while true
        nH_input = copy(nH)

        # Build the current transport operator: effective elastic frequencies,
        # total loss frequency, and sweep coefficients.
        collision_freqs = _compute_omega_values(kh, fH, nH)
        alpha_c = _compute_collisions_frequency(kh, collision_freqs, gamma_wall)
        meq_coeffs = _compute_mesh_equation_coefficients(kh, alpha_c)

        # Generation 0: direct boundary/source neutrals transported without
        # being produced by a prior collision generation.
        _debrief_msg(kh, "Computing atomic neutral generation#0", 0)
        for j in kh.vx_pos
            for i in 1:nvr
                fHG[i, j, 1] = fH[i, j, 1]
            end
        end
        # positive vx sweep: Python k=0:nx-2 -> Julia k=1:nx-1
        for k in 1:(nx - 1)
            kp1 = k + 1
            for j in kh.vx_pos
                for i in 1:nvr
                    fHG[i, j, kp1] =
                        fHG[i, j, k] * meq_coeffs.A[i, j, k] +
                        meq_coeffs.F[i, j, k]
                end
            end
        end

        # negative vx sweep: Python k=nx-1:-1:1 -> Julia k=nx:-1:2
        for k in nx:-1:2
            km1 = k - 1
            for j in kh.vx_neg
                for i in 1:nvr
                    fHG[i, j, km1] =
                        fHG[i, j, k] * meq_coeffs.C[i, j, k] +
                        meq_coeffs.G[i, j, k]
                end
            end
        end

        for k in 1:nx
            s = 0.0
            for j in 1:nvx
                dvxj = kh.dvx[j]
                for i in 1:nvr
                    s += kh.dvr_vol[i] * fHG[i, j, k] * dvxj
                end
            end
            NHG[k, 1] = s
        end

        fH = copy(fHG)
        nH = copy(NHG[:, 1])

        # Higher generations add neutrals produced by CX and elastic collision
        # source terms from the previous generation.
        fH, nH, fHG, NHG, Beta_CX_sum, m_sums, igen = _run_generations(kh, fH, nH, fHG, NHG, meq_coeffs, collision_freqs, fH_iterate)
        kh.internal.MH_H_sum = m_sums.H_H

        # Recompute density from total fH to match the reference implementation
        # after accumulating all accepted generations.
        for k in 1:nx
            s = 0.0
            for j in 1:nvx
                dvxj = kh.dvx[j]
                for i in 1:nvr
                    s += kh.dvr_vol[i] * fH[i, j, k] * dvxj
                end 
            end
            nH[k] = s
        end

        if fH_iterate
            # Outer convergence is measured by change in total neutral density
            # between fixed-point iterations.
            maxdiff = 0.0
            maxnH = 0.0

            @inbounds for k in eachindex(nH)
                maxdiff = max(maxdiff, abs(nH_input[k] - nH[k]))
                maxnH = max(maxnH, nH[k])
            end

            kh.internal.Delta_nHs = maxdiff / maxnH

            if kh.internal.Delta_nHs <= 10.0 * kh.truncate
                break
            end
        else
            break
        end
    end

    # Include the source contributions from the final generation for diagnostic
    # terms used in _compile_results.
    Beta_CX = _compute_beta_cx(kh, fHG)

    @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
        Beta_CX_sum[i, j, k] += Beta_CX[i, j, k]
    end
    m_vals = _compute_mh_values(kh, fHG, NHG[:, igen + 1])
    @inbounds for k in 1:nx, j in 1:nvx, i in 1:nvr
        m_sums.H_H[i, j, k] += m_vals.H_H[i, j, k]
        m_sums.H_P[i, j, k] += m_vals.H_P[i, j, k]
        m_sums.H_H2[i, j, k] += m_vals.H_H2[i, j, k]
    end

    return fH, nH, alpha_c, Beta_CX_sum, collision_freqs, m_sums
end

function _compile_results(
    kh::KineticH,
    fH::Array{Float64,3},
    nH::Vector{Float64},
    fSH::Array{Float64,3},
    gamma_wall::Array{Float64,3},
    alpha_c::Array{Float64,3},
    Beta_CX_sum::Array{Float64,3},
    collision_freqs::CollisionType{Vector{Float64}},
    m_sums::CollisionType{Array{Float64,3}},
    )::KHResults
    nvr, nvx, nx = kh.nvr, kh.nvx, kh.nx
    vr, vx = kh.mesh.vr, kh.mesh.vx

    # Final output construction: integrate the converged fH over velocity space
    # to obtain fluxes, temperature/pressure, source rates, and diagnostics.
    GammaxH = zeros(Float64, nx); VxH = zeros(Float64, nx); VxH_vth = zeros(Float64, nx)
    pH = zeros(Float64, nx); TH = zeros(Float64, nx); qxH = zeros(Float64, nx)
    qxH_total = zeros(Float64, nx); NetHSource = zeros(Float64, nx); Sion = zeros(Float64, nx)
    QH = zeros(Float64, nx); RxH = zeros(Float64, nx); QH_total = zeros(Float64, nx)
    SideWallH = zeros(Float64, nx)

    for out in (
        kh.output.piH_xx, kh.output.piH_yy, kh.output.piH_zz, kh.output.RxHCX,
        kh.output.RxH2_H, kh.output.RxP_H, kh.output.RxW_H, kh.output.EHCX,
        kh.output.EH2_H, kh.output.EP_H, kh.output.EW_H, kh.output.Epara_PerpH_H,
        kh.output.SourceH, kh.output.SRecomb,
    )
        fill!(out, 0.0)
    end

    # Particle flux and mean flow are needed by the random-velocity moments.
    @inbounds for k in 1:nx
        flux_sum = 0.0
        for j in 1:nvx
            vx_dvx = vx[j] * kh.dvx[j]
            for i in 1:nvr
                flux_sum += kh.dvr_vol[i] * fH[i, j, k] * vx_dvx
            end
        end
        GammaxH[k] = kh.vth * flux_sum
        VxH[k] = GammaxH[k] / nH[k]
        VxH_vth[k] = VxH[k] / kh.vth
    end

    pH_coef = kh.mu * H_MASS * kh.vth^2 / (3.0 * Q)
    piH_coef = kh.mu * H_MASS * kh.vth^2 / Q
    qxH_coef = 0.5 * kh.mu * H_MASS * kh.vth^3
    E_coef = 0.5 * kh.mu * H_MASS * kh.vth^2
    Rx_coef = kh.mu * H_MASS * kh.vth

    # Pressure tensor and heat flux moments of the final neutral distribution.
    @inbounds for k in 1:nx
        p_sum = 0.0; pi_xx_sum = 0.0; pi_yy_sum = 0.0; qx_sum = 0.0
        source_sum = 0.0; side_wall_sum = 0.0
        for j in 1:nvx
            vxran = vx[j] - VxH_vth[k]
            vxran2 = vxran^2
            dvxj = kh.dvx[j]
            for i in 1:nvr
                vr2 = vr[i]^2
                vran2 = vr2 + vxran2
                w = kh.dvr_vol[i]
                f = fH[i, j, k]
                p_sum += w * vran2 * f * dvxj
                pi_xx_sum += w * f * vxran2 * dvxj
                pi_yy_sum += w * vr2 * f * dvxj
                qx_sum += w * vran2 * f * vxran * dvxj
                source_sum += w * fSH[i, j, k] * dvxj
                side_wall_sum += w * gamma_wall[i, j, k] * f * dvxj
            end
        end
        pH[k] = pH_coef * p_sum
        TH[k] = pH[k] / nH[k]
        kh.output.piH_xx[k] = piH_coef * pi_xx_sum - pH[k]
        kh.output.piH_yy[k] = 0.5 * piH_coef * pi_yy_sum - pH[k]
        kh.output.piH_zz[k] = kh.output.piH_yy[k]
        qxH[k] = qxH_coef * qx_sum
        kh.output.SourceH[k] = source_sum
        SideWallH[k] = side_wall_sum
        kh.output.SRecomb[k] = kh.recomb ? kh.vth * kh.internal.ni[k] * kh.internal.Rec[k] : 0.0
    end

    # Net collisional/source operator C for final energy and momentum exchange
    # diagnostics. This mirrors the source-minus-sink balance used in the solve.
    @inbounds for k in 1:nx
        Q_sum = 0.0; Rx_sum = 0.0; Net_sum = 0.0
        RxHCX_sum = 0.0; EHCX_sum = 0.0; RxH2_sum = 0.0; EH2_sum = 0.0
        RxP_sum = 0.0; EP_sum = 0.0; RxW_sum = 0.0; EW_sum = 0.0; Epara_sum = 0.0
        for j in 1:nvx
            vxran = vx[j] - VxH_vth[k]
            vxran2 = vxran^2
            dvxj = kh.dvx[j]
            for i in 1:nvr
                vr2 = vr[i]^2
                f = fH[i, j, k]
                w = kh.dvr_vol[i]
                vr2vx2 = kh.internal.vr2vx2[i, j, k]
                C = kh.vth * (kh.internal.Sn[i, j, k] + Beta_CX_sum[i, j, k] - alpha_c[i, j, k] * f +
                    collision_freqs.H_P[k] * m_sums.H_P[i, j, k] +
                    collision_freqs.H_H2[k] * m_sums.H_H2[i, j, k] +
                    collision_freqs.H_H[k] * m_sums.H_H[i, j, k])
                Q_sum += w * (vr2 + vxran2) * C * dvxj
                Rx_sum += w * C * vxran * dvxj
                Net_sum += w * C * dvxj
                if kh.collisions.H_P_CX
                    CCX = kh.vth * (Beta_CX_sum[i, j, k] - kh.internal.Alpha_CX[i, j, k] * f)
                    RxHCX_sum += w * CCX * vxran * dvxj
                    EHCX_sum += w * vr2vx2 * CCX * dvxj
                end
                if kh.collisions.H2_H_EL
                    CH_H2 = kh.vth * collision_freqs.H_H2[k] * (m_sums.H_H2[i, j, k] - f)
                    RxH2_sum += w * CH_H2 * vxran * dvxj
                    EH2_sum += w * vr2vx2 * CH_H2 * dvxj
                end
                if kh.collisions.H_P_EL
                    CH_P = kh.vth * collision_freqs.H_P[k] * (m_sums.H_P[i, j, k] - f)
                    RxP_sum += w * CH_P * vxran * dvxj
                    EP_sum += w * vr2vx2 * CH_P * dvxj
                end
                CW_H = -kh.vth * gamma_wall[i, j, k] * f
                RxW_sum += w * CW_H * vxran * dvxj
                EW_sum += w * vr2vx2 * CW_H * dvxj
                if kh.collisions.H_H_EL
                    CH_H = kh.vth * collision_freqs.H_H[k] * (m_sums.H_H[i, j, k] - f)
                    Epara_sum += w * (vr2 - 2.0 * vxran2) * CH_H * dvxj
                end
            end
        end
        QH[k] = E_coef * Q_sum
        RxH[k] = Rx_coef * Rx_sum
        NetHSource[k] = Net_sum
        Sion[k] = kh.vth * nH[k] * kh.internal.alpha_ion[k]
        kh.output.RxHCX[k] = Rx_coef * RxHCX_sum
        kh.output.EHCX[k] = E_coef * EHCX_sum
        kh.output.RxH2_H[k] = Rx_coef * RxH2_sum
        kh.output.EH2_H[k] = E_coef * EH2_sum
        kh.output.RxP_H[k] = Rx_coef * RxP_sum
        kh.output.EP_H[k] = E_coef * EP_sum
        kh.output.RxW_H[k] = Rx_coef * RxW_sum
        kh.output.EW_H[k] = E_coef * EW_sum
        kh.output.Epara_PerpH_H[k] = -E_coef * Epara_sum
        qxH_total[k] = (0.5 * nH[k] * kh.mu * H_MASS * VxH[k]^2 + 2.5 * pH[k] * Q) * VxH[k] +
                       Q * kh.output.piH_xx[k] * VxH[k] + qxH[k]
        QH_total[k] = QH[k] + RxH[k] * VxH[k] + 0.5 * kh.mu * H_MASS * NetHSource[k] * VxH[k]^2
    end

    # Albedo is the ratio of outgoing negative-vx flux to incoming positive-vx
    # flux at the left boundary.
    gammax_plus = 0.0; gammax_minus = 0.0
    @inbounds for j in kh.vx_pos, i in 1:nvr
        gammax_plus += kh.dvr_vol[i] * fH[i, j, 1] * vx[j] * kh.dvx[j]
    end
    @inbounds for j in kh.vx_neg, i in 1:nvr
        gammax_minus += kh.dvr_vol[i] * fH[i, j, 1] * vx[j] * kh.dvx[j]
    end
    gammax_plus *= kh.vth
    gammax_minus *= kh.vth
    AlbedoH = abs(gammax_plus) > 0.0 ? -gammax_minus / gammax_plus : 0.0

    return KHResults(fH, nH, GammaxH, VxH, pH, TH, qxH, qxH_total, NetHSource, Sion, QH, RxH, QH_total, AlbedoH, SideWallH)
end
