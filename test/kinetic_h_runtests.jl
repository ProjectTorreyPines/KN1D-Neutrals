using Test
using KN1DJl

function _sample_array3(A::Array{Float64,3})
    n1, n2, n3 = size(A)
    return [
        A[1, 1, 1],
        A[min(2, n1), min(3, n2), min(4, n3)],
        A[fld(n1 - 1, 2) + 1, fld(n2 - 1, 2) + 1, fld(n3 - 1, 2) + 1],
        A[n1, n2, n3],
    ]
end

function _sample_vector(v::AbstractVector{<:Real})
    n = length(v)
    return Float64[v[1], v[fld(n - 1, 2) + 1], v[n]]
end

const KINETIC_H_TEST_CONFIG = joinpath(@__DIR__, "kinetic_h_test_config.json")

function _write_kinetic_h_test_config()::Nothing
    open(KINETIC_H_TEST_CONFIG, "w") do io
        JSON.print(io, Dict(
            "kinetic_h" => Dict(
                "mesh_size" => 10,
                "ion_rate" => "janev",
                "ci_test" => false,
                "alpha_cx_test" => false,
                "grid_fctr" => 0.3,
                "extra_energy_bins_eV" => Float64[],
            ),
            "kinetic_h2" => Dict(
                "mesh_size" => 6,
                "grid_fctr" => 0.3,
                "extra_energy_bins_eV" => Float64[],
                "ci_test" => false,
                "alpha_cx_test" => false,
            ),
            "collisions" => Dict(
                "H2_H_EL" => true,
                "H2_H2_EL" => true,
                "H2_P_EL" => true,
                "H2_P_CX" => true,
                "H_H_EL" => true,
                "H_P_EL" => true,
                "H_P_CX" => true,
                "SIMPLE_CX" => true,
            ),
        ))
    end
    return nothing
end

function _kinetic_h_fixture(config_path::AbstractString)
    x = collect(range(0.0, 1.0; length=5))
    Ti = collect(range(1.5, 2.5; length=5))
    Te = collect(range(2.0, 4.0; length=5))
    ne = collect(range(8.0e17, 1.2e18; length=5))
    pipe = fill(0.12, 5)
    mesh = KineticMesh("h", 1, x, Ti, Te, ne, pipe; config_path=String(config_path))
    fbc = zeros(Float64, length(mesh.vr), length(mesh.vx))
    @inbounds for j in eachindex(mesh.vx), i in eachindex(mesh.vr)
        if mesh.vx[j] > 0.0
            fbc[i, j] = 0.75 + 0.02 * i + 0.01 * j
        end
    end
    kh = KineticH(
        mesh,
        1,
        collect(range(-150.0, 150.0; length=length(mesh.x))),
        fbc,
        1.0e20;
        config_path=String(config_path),
        initialize_static=true,
    )

    fH = zeros(Float64, kh.nvr, kh.nvx, kh.nx)
    fH2 = Array{Float64,3}(undef, kh.nvr, kh.nvx, kh.nx)
    fSH = Array{Float64,3}(undef, kh.nvr, kh.nvx, kh.nx)
    @inbounds for k in 1:kh.nx, j in 1:kh.nvx, i in 1:kh.nvr
        fH2[i, j, k] = 1.0e10 * (1.0 + 0.01 * i + 0.002 * j + 0.0001 * k)
        fSH[i, j, k] = 1.0e16 * (1.0 + 0.003 * i + 0.001 * j + 0.0002 * k)
    end
    nHP = collect(range(1.0e16, 2.0e16; length=kh.nx))
    THP = collect(range(1.0, 1.5; length=kh.nx))

    return kh, fH, fH2, fSH, nHP, THP
end

@testset "KineticH Static And Dynamic Internals Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KineticH parity tests require python with numpy/scipy"
    else
        _write_kinetic_h_test_config()
        config_literal = JSON.json(KINETIC_H_TEST_CONFIG)
        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kinetic_mesh import KineticMesh
            from KN1DPy.kinetic_h import KineticH
            config_path = $config_literal

            x = np.linspace(0.0, 1.0, 5)
            Ti = np.linspace(1.5, 2.5, 5)
            Te = np.linspace(2.0, 4.0, 5)
            ne = np.linspace(8.0e17, 1.2e18, 5)
            pipe = np.full(5, 0.12)
            mesh = KineticMesh('h', 1, x, Ti, Te, ne, pipe, config_path=config_path)

            fbc = np.zeros((len(mesh.vr), len(mesh.vx)))
            for i in range(len(mesh.vr)):
                for j in range(len(mesh.vx)):
                    if mesh.vx[j] > 0.0:
                        fbc[i, j] = 0.75 + 0.02 * (i + 1) + 0.01 * (j + 1)

            kh = KineticH(
                mesh,
                1,
                np.linspace(-150.0, 150.0, len(mesh.x)),
                fbc,
                1.0e20,
                config_path=config_path,
            )

            fH = np.zeros((kh.nvr, kh.nvx, kh.nx))
            fH2 = np.empty((kh.nvr, kh.nvx, kh.nx))
            fSH = np.empty((kh.nvr, kh.nvx, kh.nx))
            for k in range(kh.nx):
                for j in range(kh.nvx):
                    for i in range(kh.nvr):
                        fH2[i, j, k] = 1.0e10 * (1.0 + 0.01 * (i + 1) + 0.002 * (j + 1) + 0.0001 * (k + 1))
                        fSH[i, j, k] = 1.0e16 * (1.0 + 0.003 * (i + 1) + 0.001 * (j + 1) + 0.0002 * (k + 1))

            nHP = np.linspace(1.0e16, 2.0e16, kh.nx)
            THP = np.linspace(1.0, 1.5, kh.nx)
            kh._compute_dynamic_internals(fH, fH2, nHP, THP, fSH)

            def sample3(A):
                n1, n2, n3 = A.shape
                return [
                    float(A[0, 0, 0]),
                    float(A[min(1, n1-1), min(2, n2-1), min(3, n3-1)]),
                    float(A[(n1-1)//2, (n2-1)//2, (n3-1)//2]),
                    float(A[-1, -1, -1]),
                ]

            print(json.dumps({
                "size": [kh.nvr, kh.nvx, kh.nx],
                "vr2vx2": sample3(kh.Internal.vr2vx2),
                "vr2vx_vxi2": sample3(kh.Internal.vr2vx_vxi2),
                "ErelH_P": sample3(kh.Internal.ErelH_P),
                "fi_hat": sample3(kh.Internal.fi_hat),
                "Ti_mu": sample3(kh.Internal.Ti_mu),
                "sigv_sum": float(np.sum(kh.Internal.sigv)),
                "alpha_ion": [float(kh.Internal.alpha_ion[0]), float(kh.Internal.alpha_ion[len(kh.Internal.alpha_ion)//2]), float(kh.Internal.alpha_ion[-1])],
                "Rec": [float(kh.Internal.Rec[0]), float(kh.Internal.Rec[len(kh.Internal.Rec)//2]), float(kh.Internal.Rec[-1])],
                "ni": [float(kh.Internal.ni[0]), float(kh.Internal.ni[len(kh.Internal.ni)//2]), float(kh.Internal.ni[-1])],
                "Sn": sample3(kh.Internal.Sn),
                "nH2": [float(kh.H2_Moments.nH2[0]), float(kh.H2_Moments.nH2[len(kh.H2_Moments.nH2)//2]), float(kh.H2_Moments.nH2[-1])],
                "VxH2": [float(kh.H2_Moments.VxH2[0]), float(kh.H2_Moments.VxH2[len(kh.H2_Moments.VxH2)//2]), float(kh.H2_Moments.VxH2[-1])],
                "TH2": [float(kh.H2_Moments.TH2[0]), float(kh.H2_Moments.TH2[len(kh.H2_Moments.TH2)//2]), float(kh.H2_Moments.TH2[-1])],
                "Alpha_CX": sample3(kh.Internal.Alpha_CX),
                "Alpha_H_H2": sample3(kh.Internal.Alpha_H_H2),
                "Alpha_H_P": sample3(kh.Internal.Alpha_H_P),
                "Alpha_CX_sum": float(np.sum(kh.Internal.Alpha_CX)),
                "Alpha_H_H2_sum": float(np.sum(kh.Internal.Alpha_H_H2)),
                "Alpha_H_P_sum": float(np.sum(kh.Internal.Alpha_H_P)),
            }))
            """
        )

        kh, fH, fH2, fSH, nHP, THP = _kinetic_h_fixture(KINETIC_H_TEST_CONFIG)
        KN1DJl._compute_dynamic_internals!(kh, fH, fH2, nHP, THP, fSH)

        @test [kh.nvr, kh.nvx, kh.nx] == Int.(py["size"])
        @test _sample_array3(kh.internal.vr2vx2) ≈ _to_float_vector(py["vr2vx2"]) atol=1e-12 rtol=1e-12
        # The KineticMesh x grid has tiny Julia/Python interpolation differences; these propagate here.
        @test _sample_array3(kh.internal.vr2vx_vxi2) ≈ _to_float_vector(py["vr2vx_vxi2"]) atol=1e-7 rtol=1e-6
        @test _sample_array3(kh.internal.ErelH_P) ≈ _to_float_vector(py["ErelH_P"]) atol=1e-7 rtol=1e-6
        @test _sample_array3(kh.internal.fi_hat) ≈ _to_float_vector(py["fi_hat"]) atol=1e-12 rtol=1e-10
        @test _sample_array3(kh.internal.Ti_mu) ≈ _to_float_vector(py["Ti_mu"]) atol=1e-7 rtol=1e-6
        @test sum(kh.internal.sigv) ≈ Float64(py["sigv_sum"]) atol=1e-30 rtol=1e-6
        midx = fld(kh.nx, 2) + 1
        @test kh.internal.alpha_ion[[1, midx, kh.nx]] ≈ _to_float_vector(py["alpha_ion"]) atol=1e-30 rtol=1e-6
        @test kh.internal.Rec[[1, midx, kh.nx]] ≈ _to_float_vector(py["Rec"]) atol=1e-30 rtol=1e-6
        @test kh.internal.ni[[1, midx, kh.nx]] ≈ _to_float_vector(py["ni"]) atol=1e-6 rtol=1e-8
        @test _sample_array3(kh.internal.Sn) ≈ _to_float_vector(py["Sn"]) atol=1e-6 rtol=1e-6
        @test kh.h2_moments.nH2[[1, midx, kh.nx]] ≈ _to_float_vector(py["nH2"]) atol=1e-6 rtol=1e-10
        @test kh.h2_moments.VxH2[[1, midx, kh.nx]] ≈ _to_float_vector(py["VxH2"]) atol=1e-8 rtol=1e-6
        @test kh.h2_moments.TH2[[1, midx, kh.nx]] ≈ _to_float_vector(py["TH2"]) atol=1e-8 rtol=1e-6
        @test _sample_array3(kh.internal.Alpha_CX) ≈ _to_float_vector(py["Alpha_CX"]) atol=1e-10 rtol=1e-6
        @test _sample_array3(kh.internal.Alpha_H_H2) ≈ _to_float_vector(py["Alpha_H_H2"]) atol=1e-12 rtol=1e-10
        @test _sample_array3(kh.internal.Alpha_H_P) ≈ _to_float_vector(py["Alpha_H_P"]) atol=1e-10 rtol=1e-6
        @test sum(kh.internal.Alpha_CX) ≈ Float64(py["Alpha_CX_sum"]) atol=1e-8 rtol=1e-6
        @test sum(kh.internal.Alpha_H_H2) ≈ Float64(py["Alpha_H_H2_sum"]) atol=1e-12 rtol=1e-10
        @test sum(kh.internal.Alpha_H_P) ≈ Float64(py["Alpha_H_P_sum"]) atol=1e-8 rtol=1e-6
    end
end

@testset "KineticH Solver Helper Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KineticH solver helper parity requires python with numpy/scipy"
    else
        _write_kinetic_h_test_config()
        config_literal = JSON.json(KINETIC_H_TEST_CONFIG)
        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kinetic_mesh import KineticMesh
            from KN1DPy.kinetic_h import KineticH
            config_path = $config_literal

            x = np.linspace(0.0, 1.0, 5)
            Ti = np.linspace(1.5, 2.5, 5)
            Te = np.linspace(2.0, 4.0, 5)
            ne = np.linspace(8.0e17, 1.2e18, 5)
            pipe = np.full(5, 0.12)
            mesh = KineticMesh('h', 1, x, Ti, Te, ne, pipe, config_path=config_path)

            fbc = np.zeros((len(mesh.vr), len(mesh.vx)))
            for i in range(len(mesh.vr)):
                for j in range(len(mesh.vx)):
                    if mesh.vx[j] > 0.0:
                        fbc[i, j] = 0.75 + 0.02 * (i + 1) + 0.01 * (j + 1)

            kh = KineticH(
                mesh,
                1,
                np.linspace(-150.0, 150.0, len(mesh.x)),
                fbc,
                1.0e20,
                config_path=config_path,
            )

            fH = np.empty((kh.nvr, kh.nvx, kh.nx))
            fH2 = np.empty((kh.nvr, kh.nvx, kh.nx))
            fSH = np.empty((kh.nvr, kh.nvx, kh.nx))
            for k in range(kh.nx):
                for j in range(kh.nvx):
                    for i in range(kh.nvr):
                        fH[i, j, k] = 1.0e11 * (1.0 + 0.015 * (i + 1) + 0.004 * (j + 1) + 0.0007 * (k + 1))
                        fH2[i, j, k] = 1.0e10 * (1.0 + 0.01 * (i + 1) + 0.002 * (j + 1) + 0.0001 * (k + 1))
                        fSH[i, j, k] = 1.0e16 * (1.0 + 0.003 * (i + 1) + 0.001 * (j + 1) + 0.0002 * (k + 1))

            nHP = np.linspace(1.0e16, 2.0e16, kh.nx)
            THP = np.linspace(1.0, 1.5, kh.nx)
            kh._compute_dynamic_internals(fH, fH2, nHP, THP, fSH)

            nH = np.zeros(kh.nx)
            for k in range(kh.nx):
                nH[k] = np.sum(kh.dvr_vol * (fH[:, :, k] @ kh.dvx))

            gamma_wall = np.zeros((kh.nvr, kh.nvx, kh.nx))
            for k in range(kh.nx):
                if kh.mesh.PipeDia[k] > 0:
                    gamma_wall[:, :, k] = (2 * kh.mesh.vr / kh.mesh.PipeDia[k])[:, None]

            omega = kh._compute_omega_values(fH, nH)
            alpha_c = kh._compute_collision_frequency(omega, gamma_wall)
            meq = kh._compute_mesh_equation_coefficients(alpha_c)
            beta = kh._compute_beta_cx(fH)
            mh = kh._compute_mh_values(fH, nH)

            def sample1(v):
                return [float(v[0]), float(v[(len(v)-1)//2]), float(v[-1])]

            def sample3(A):
                n1, n2, n3 = A.shape
                return [
                    float(A[0, 0, 0]),
                    float(A[min(1, n1-1), min(2, n2-1), min(3, n3-1)]),
                    float(A[(n1-1)//2, (n2-1)//2, (n3-1)//2]),
                    float(A[-1, -1, -1]),
                ]

            print(json.dumps({
                "nH": sample1(nH),
                "omega_H_H": sample1(omega.H_H),
                "omega_H_P": sample1(omega.H_P),
                "omega_H_H2": sample1(omega.H_H2),
                "omega_sums": [float(np.sum(omega.H_H)), float(np.sum(omega.H_P)), float(np.sum(omega.H_H2))],
                "alpha_c": sample3(alpha_c),
                "alpha_c_sum": float(np.sum(alpha_c)),
                "meq_A": sample3(meq.A),
                "meq_B": sample3(meq.B),
                "meq_C": sample3(meq.C),
                "meq_D": sample3(meq.D),
                "meq_F": sample3(meq.F),
                "meq_G": sample3(meq.G),
                "meq_sums": [float(np.sum(meq.A)), float(np.sum(meq.B)), float(np.sum(meq.C)), float(np.sum(meq.D)), float(np.sum(meq.F)), float(np.sum(meq.G))],
                "beta": sample3(beta),
                "beta_sum": float(np.sum(beta)),
                "mh_H_H": sample3(mh.H_H),
                "mh_H_P": sample3(mh.H_P),
                "mh_H_H2": sample3(mh.H_H2),
                "mh_sums": [float(np.sum(mh.H_H)), float(np.sum(mh.H_P)), float(np.sum(mh.H_H2))],
                "max_dx": sample1(kh.Errors.Max_dx),
            }))
            """
        )

        kh, fH, fH2, fSH, nHP, THP = _kinetic_h_fixture(KINETIC_H_TEST_CONFIG)
        @inbounds for k in 1:kh.nx, j in 1:kh.nvx, i in 1:kh.nvr
            fH[i, j, k] = 1.0e11 * (1.0 + 0.015 * i + 0.004 * j + 0.0007 * k)
        end
        KN1DJl._compute_dynamic_internals!(kh, fH, fH2, nHP, THP, fSH)

        nH = zeros(Float64, kh.nx)
        @inbounds for k in 1:kh.nx
            s = 0.0
            for j in 1:kh.nvx
                for i in 1:kh.nvr
                    s += kh.dvr_vol[i] * fH[i, j, k] * kh.dvx[j]
                end
            end
            nH[k] = s
        end

        gamma_wall = zeros(Float64, kh.nvr, kh.nvx, kh.nx)
        @inbounds for k in 1:kh.nx
            if kh.mesh.PipeDia[k] > 0.0
                for j in 1:kh.nvx, i in 1:kh.nvr
                    gamma_wall[i, j, k] = 2.0 * kh.mesh.vr[i] / kh.mesh.PipeDia[k]
                end
            end
        end

        omega = KN1DJl._compute_omega_values(kh, fH, nH)
        alpha_c = KN1DJl._compute_collision_frequency(kh, omega, gamma_wall)
        meq = KN1DJl._compute_mesh_equation_coefficients(kh, alpha_c)
        beta = KN1DJl._compute_beta_cx(kh, fH)
        mh = KN1DJl._compute_mh_values(kh, fH, nH)

        @test _sample_vector(nH) ≈ _to_float_vector(py["nH"]) atol=1e-6 rtol=1e-10
        @test _sample_vector(omega.H_H) ≈ _to_float_vector(py["omega_H_H"]) atol=1e-12 rtol=1e-6
        @test _sample_vector(omega.H_P) ≈ _to_float_vector(py["omega_H_P"]) atol=1e-12 rtol=1e-6
        @test _sample_vector(omega.H_H2) ≈ _to_float_vector(py["omega_H_H2"]) atol=1e-12 rtol=1e-6
        @test [sum(omega.H_H), sum(omega.H_P), sum(omega.H_H2)] ≈ _to_float_vector(py["omega_sums"]) atol=1e-12 rtol=1e-6
        @test _sample_array3(alpha_c) ≈ _to_float_vector(py["alpha_c"]) atol=1e-8 rtol=1e-6
        @test sum(alpha_c) ≈ Float64(py["alpha_c_sum"]) atol=1e-6 rtol=1e-6
        @test _sample_array3(meq.A) ≈ _to_float_vector(py["meq_A"]) atol=1e-8 rtol=1e-6
        @test _sample_array3(meq.B) ≈ _to_float_vector(py["meq_B"]) atol=1e-12 rtol=1e-6
        @test _sample_array3(meq.C) ≈ _to_float_vector(py["meq_C"]) atol=1e-8 rtol=1e-6
        @test _sample_array3(meq.D) ≈ _to_float_vector(py["meq_D"]) atol=1e-12 rtol=1e-6
        @test _sample_array3(meq.F) ≈ _to_float_vector(py["meq_F"]) atol=1e-2 rtol=1e-6
        @test _sample_array3(meq.G) ≈ _to_float_vector(py["meq_G"]) atol=1e-2 rtol=1e-6
        @test [sum(meq.A), sum(meq.B), sum(meq.C), sum(meq.D), sum(meq.F), sum(meq.G)] ≈ _to_float_vector(py["meq_sums"]) atol=1e-2 rtol=1e-6
        @test _sample_array3(beta) ≈ _to_float_vector(py["beta"]) atol=1e-8 rtol=1e-6
        @test sum(beta) ≈ Float64(py["beta_sum"]) atol=1e-6 rtol=1e-6
        @test _sample_array3(mh.H_H) ≈ _to_float_vector(py["mh_H_H"]) atol=1e-6 rtol=1e-6
        @test _sample_array3(mh.H_P) ≈ _to_float_vector(py["mh_H_P"]) atol=1e-6 rtol=1e-6
        @test _sample_array3(mh.H_H2) ≈ _to_float_vector(py["mh_H_H2"]) atol=1e-6 rtol=1e-6
        @test [sum(mh.H_H), sum(mh.H_P), sum(mh.H_H2)] ≈ _to_float_vector(py["mh_sums"]) atol=1e-4 rtol=1e-6
        @test _sample_vector(kh.errors.Max_dx) ≈ _to_float_vector(py["max_dx"]) atol=1e-8 rtol=1e-6
    end
end

@testset "KineticH run_procedure Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KineticH run_procedure parity requires python with numpy/scipy"
    else
        _write_kinetic_h_test_config()
        config_literal = JSON.json(KINETIC_H_TEST_CONFIG)
        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kinetic_mesh import KineticMesh
            from KN1DPy.kinetic_h import KineticH
            config_path = $config_literal
            x = np.linspace(0.0, 0.05, 5)
            mesh = KineticMesh('h', 1, x, np.full(5, 2.0), np.full(5, 3.0), np.full(5, 1.0e18), np.zeros(5), config_path=config_path)
            fbc = np.zeros((len(mesh.vr), len(mesh.vx)))
            fbc[:, mesh.vx > 0] = 1.0
            kh = KineticH(mesh, 1, np.zeros(len(mesh.x)), fbc, 1.0e20, truncate=1.0, max_gen=3, config_path=config_path)
            r = kh.run_procedure()
            print(json.dumps({
                "nH": [float(r.nH[0]), float(r.nH[len(r.nH)//2]), float(r.nH[-1])],
                "GammaxH": [float(r.GammaxH[0]), float(r.GammaxH[len(r.GammaxH)//2]), float(r.GammaxH[-1])],
                "TH": [float(r.TH[0]), float(r.TH[len(r.TH)//2]), float(r.TH[-1])],
                "Sion": [float(r.Sion[0]), float(r.Sion[len(r.Sion)//2]), float(r.Sion[-1])],
                "AlbedoH": float(r.AlbedoH),
                "fH_sum": float(np.sum(r.fH)),
            }))
            """
        )

        x = collect(range(0.0, 0.05; length=5))
        mesh = KineticMesh("h", 1, x, fill(2.0, 5), fill(3.0, 5), fill(1.0e18, 5), zeros(5); config_path=KINETIC_H_TEST_CONFIG)
        fbc = zeros(Float64, length(mesh.vr), length(mesh.vx))
        fbc[:, mesh.vx .> 0.0] .= 1.0
        kh = KineticH(mesh, 1, zeros(length(mesh.x)), fbc, 1.0e20;
            truncate=1.0, max_gen=3, config_path=KINETIC_H_TEST_CONFIG, initialize_static=true)
        r = run_procedure(kh)
        midx = fld(length(r.nH), 2) + 1

        @test r.nH[[1, midx, end]] ≈ _to_float_vector(py["nH"]) rtol=5e-4 atol=1e-8
        @test r.GammaxH[[1, midx, end]] ≈ _to_float_vector(py["GammaxH"]) rtol=5e-4 atol=1e-8
        @test r.TH[[1, midx, end]] ≈ _to_float_vector(py["TH"]) rtol=5e-4 atol=1e-8
        @test r.Sion[[1, midx, end]] ≈ _to_float_vector(py["Sion"]) rtol=5e-4 atol=1e-8
        @test r.AlbedoH ≈ Float64(py["AlbedoH"]) rtol=5e-4 atol=1e-8
        @test sum(r.fH) ≈ Float64(py["fH_sum"]) rtol=5e-4 atol=1e-8
    end
end
