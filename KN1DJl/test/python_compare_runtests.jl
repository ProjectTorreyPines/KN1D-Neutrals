using Test
using JSON
using KN1DJl

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PYTHON = let
    override = get(ENV, "KN1DJL_PYTHON", "")
    if !isempty(override)
        override
    else
        venv_python = joinpath(REPO_ROOT, ".venv", "bin", "python")
        isfile(venv_python) ? venv_python : "python3"
    end
end

function _python_numpy_available()::Bool
    try
        run(pipeline(setenv(`$(PYTHON) -c "import numpy"`, "PYTHONPATH" => REPO_ROOT); stdout=devnull, stderr=devnull))
        return true
    catch
        return false
    end
end

function _python_json(script::AbstractString)
    cmd = `$(PYTHON) -c $script`
    raw = read(setenv(cmd, "PYTHONPATH" => REPO_ROOT), String)
    lines = filter(!isempty, strip.(split(raw, '\n')))
    isempty(lines) && error("Python command produced no output")
    return JSON.parse(lines[end])
end

function _to_float_vector(x)
    return Float64[v for v in x]
end

function _to_float_matrix(x)
    rows = length(x)
    cols = rows == 0 ? 0 : length(x[1])
    A = Matrix{Float64}(undef, rows, cols)
    @inbounds for i in 1:rows, j in 1:cols
        A[i, j] = x[i][j]
    end
    return A
end

function _to_float_array3(x)
    n1 = length(x)
    n2 = n1 == 0 ? 0 : length(x[1])
    n3 = (n1 == 0 || n2 == 0) ? 0 : length(x[1][1])
    A = Array{Float64,3}(undef, n1, n2, n3)
    @inbounds for i in 1:n1, j in 1:n2, k in 1:n3
        A[i, j, k] = x[i][j][k]
    end
    return A
end

@testset "Python Comparison" begin
    if !_python_numpy_available()
        @test_skip "Python comparison tests require python3 with numpy installed"
    else
    @testset "create_vr_vx_mesh parity" begin
        Ti = [1.0, 2.0, 3.0]
        E0 = [0.0, 1.5]

        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kinetic_mesh import KineticMesh
            vx, vr, Tnorm = KineticMesh.create_vr_vx_mesh(None, 4, np.array([1.0,2.0,3.0]), E0=np.array([0.0,1.5]))
            print(json.dumps({"vx": vx.tolist(), "vr": vr.tolist(), "Tnorm": float(Tnorm)}))
            """
        )

        j_vx, j_vr, j_Tnorm = create_vr_vx_mesh(4, Ti; E0=E0)

        @test j_Tnorm ≈ Float64(py["Tnorm"]) atol=0 rtol=0
        @test j_vr ≈ _to_float_vector(py["vr"]) atol=0 rtol=0
        @test j_vx ≈ _to_float_vector(py["vx"]) atol=0 rtol=0
    end

    @testset "VSpaceDifferentials parity" begin
        vr = [0.5, 1.0]
        vx = [-1.0, -0.5, 0.5, 1.0]

        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.make_dvr_dvx import VSpace_Differentials
            vsd = VSpace_Differentials(np.array([0.5, 1.0]), np.array([-1.0, -0.5, 0.5, 1.0]))
            print(json.dumps({
                "dvr_vol": vsd.dvr_vol.tolist(),
                "dvx": vsd.dvx.tolist(),
                "volume": vsd.volume.tolist(),
                "vmag_squared": vsd.vmag_squared.tolist(),
                "vx_pos_start": int(vsd.vx_pos_start),
                "vx_pos_end": int(vsd.vx_pos_end),
                "vx_neg_start": int(vsd.vx_neg_start),
                "vx_neg_end": int(vsd.vx_neg_end)
            }))
            """
        )

        j = VSpaceDifferentials(vr, vx)

        @test j.dvr_vol ≈ _to_float_vector(py["dvr_vol"]) atol=1e-14 rtol=1e-14
        @test j.dvx ≈ _to_float_vector(py["dvx"]) atol=1e-14 rtol=1e-14
        @test j.volume ≈ _to_float_matrix(py["volume"]) atol=1e-14 rtol=1e-14
        @test j.vmag_squared ≈ _to_float_matrix(py["vmag_squared"]) atol=0 rtol=0
        @test j.vx_pos_start == Int(py["vx_pos_start"]) + 1
        @test j.vx_pos_end == Int(py["vx_pos_end"]) + 1
        @test j.vx_neg_start == Int(py["vx_neg_start"]) + 1
        @test j.vx_neg_end == Int(py["vx_neg_end"]) + 1
    end

    @testset "create_shifted_maxwellian parity" begin
        vr = [0.5, 1.0]
        vx = [-1.0, -0.5, 0.5, 1.0]
        Tmaxwell = [1.0, 2.0]
        vx_shift = [0.2, 0.3]

        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.create_shifted_maxwellian import create_shifted_maxwellian
            M = create_shifted_maxwellian(
                np.array([0.5, 1.0]),
                np.array([-1.0, -0.5, 0.5, 1.0]),
                np.array([1.0, 2.0]),
                np.array([0.2, 0.3]),
                1, 1, 1.0
            )
            print(json.dumps({"M": M.tolist()}))
            """
        )

        j = create_shifted_maxwellian(vr, vx, Tmaxwell, vx_shift, 1, 1, 1.0)
        @test j ≈ _to_float_array3(py["M"]) atol=1e-12 rtol=1e-10
    end

    @testset "KineticMesh parity" begin
        x = collect(range(0.0, 1.0; length=5))
        Ti = fill(2.0, 5)
        Te = fill(3.0, 5)
        n = fill(1.0e18, 5)
        PipeDia = fill(0.1, 5)

        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kinetic_mesh import KineticMesh
            x = np.linspace(0.0, 1.0, 5)
            Ti = np.full(5, 2.0)
            Te = np.full(5, 3.0)
            n = np.full(5, 1.0e18)
            PipeDia = np.full(5, 0.1)
            km = KineticMesh('h', 1, x, Ti, Te, n, PipeDia, config_path='KN1DJl/config.json')
            print(json.dumps({
                "nx": len(km.x),
                "nvr": len(km.vr),
                "nvx": len(km.vx),
                "Tnorm": float(km.Tnorm),
                "x_head": km.x[:5].tolist(),
                "x_tail": km.x[-5:].tolist()
            }))
            """
        )

        km = KineticMesh("h", 1, x, Ti, Te, n, PipeDia; config_path=joinpath(REPO_ROOT, "KN1DJl", "config.json"))

        @test length(km.x) == Int(py["nx"])
        @test length(km.vr) == Int(py["nvr"])
        @test length(km.vx) == Int(py["nvx"])
        @test km.Tnorm ≈ Float64(py["Tnorm"]) atol=0 rtol=0
        @test km.x[1:5] ≈ _to_float_vector(py["x_head"]) atol=1e-8 rtol=1e-8
        @test km.x[end-4:end] ≈ _to_float_vector(py["x_tail"]) atol=1e-8 rtol=1e-8
    end
    end
end
