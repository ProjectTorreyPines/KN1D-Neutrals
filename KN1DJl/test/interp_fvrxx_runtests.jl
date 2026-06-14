using Test
using JSON
using KN1DJl

const INTERP_REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const INTERP_PYTHON = let
    override = get(ENV, "KN1DJL_PYTHON", "")
    if !isempty(override)
        override
    else
        venv_python = joinpath(INTERP_REPO_ROOT, ".venv", "bin", "python")
        isfile(venv_python) ? venv_python : "python3"
    end
end

function _interp_python_available()::Bool
    try
        run(pipeline(
            setenv(`$(INTERP_PYTHON) -c "import numpy, scipy"`, "PYTHONPATH" => INTERP_REPO_ROOT);
            stdout=devnull,
            stderr=devnull,
        ))
        return true
    catch
        return false
    end
end

function _interp_python_json(script::AbstractString)
    raw = read(setenv(`$(INTERP_PYTHON) -c $script`, "PYTHONPATH" => INTERP_REPO_ROOT), String)
    lines = filter(!isempty, strip.(split(raw, '\n')))
    isempty(lines) && error("Python command produced no output")
    return JSON.parse(lines[end])
end

_fvec(x) = Float64[v for v in x]

function _fmat(x)
    rows = length(x)
    cols = rows == 0 ? 0 : length(x[1])
    A = Matrix{Float64}(undef, rows, cols)
    @inbounds for i in 1:rows, j in 1:cols
        A[i, j] = Float64(x[i][j])
    end
    return A
end

function _farray3(x)
    n1 = length(x)
    n2 = n1 == 0 ? 0 : length(x[1])
    n3 = (n1 == 0 || n2 == 0) ? 0 : length(x[1][1])
    A = Array{Float64,3}(undef, n1, n2, n3)
    @inbounds for i in 1:n1, j in 1:n2, k in 1:n3
        A[i, j, k] = Float64(x[i][j][k])
    end
    return A
end

function _interp_fixture()
    mesh_a = KineticMesh(
        "fixture",
        [0.0, 0.5, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [1.0e18, 1.0e18, 1.0e18],
        [0.1, 0.1, 0.1],
        [-1.0, -0.5, 0.5, 1.0],
        [0.5, 1.0],
        1.0,
    )
    mesh_b = KineticMesh(
        "fixture",
        [0.25, 0.75],
        [2.0, 2.0],
        [3.0, 3.0],
        [1.0e18, 1.0e18],
        [0.1, 0.1],
        [-0.75, -0.25, 0.25, 0.75],
        [0.5, 0.9],
        1.0,
    )
    fa = Array{Float64,3}(undef, 2, 4, 3)
    @inbounds for k in 1:3, j in 1:4, i in 1:2
        fa[i, j, k] = 0.05 + 0.01 * i + 0.02 * j + 0.03 * k
    end
    return fa, mesh_a, mesh_b
end

const INTERP_PY_STAGE_SCRIPT = """
import sys, json, numpy as np
from types import SimpleNamespace
sys.path.insert(0, '.')
from KN1DPy.interp_fvrvxx import interp_fvrvxx, _get_interpolation_bounds
from KN1DPy.make_dvr_dvx import VSpace_Differentials

mesh_a = SimpleNamespace(
    x=np.array([0.0, 0.5, 1.0]),
    Ti=np.array([2.0, 2.0, 2.0]),
    Te=np.array([3.0, 3.0, 3.0]),
    ne=np.array([1.0e18, 1.0e18, 1.0e18]),
    PipeDia=np.array([0.1, 0.1, 0.1]),
    vx=np.array([-1.0, -0.5, 0.5, 1.0]),
    vr=np.array([0.5, 1.0]),
    Tnorm=1.0,
)
mesh_b = SimpleNamespace(
    x=np.array([0.25, 0.75]),
    Ti=np.array([2.0, 2.0]),
    Te=np.array([3.0, 3.0]),
    ne=np.array([1.0e18, 1.0e18]),
    PipeDia=np.array([0.1, 0.1]),
    vx=np.array([-0.75, -0.25, 0.25, 0.75]),
    vr=np.array([0.5, 0.9]),
    Tnorm=1.0,
)
fa = np.empty((2, 4, 3), dtype=float)
for k in range(3):
    for j in range(4):
        for i in range(2):
            fa[i, j, k] = 0.05 + 0.01 * (i + 1) + 0.02 * (j + 1) + 0.03 * (k + 1)

v_scale = np.sqrt(mesh_b.Tnorm / mesh_a.Tnorm)
vdiff_a = VSpace_Differentials(mesh_a.vr, mesh_a.vx)
vdiff_b = VSpace_Differentials(mesh_b.vr, mesh_b.vx)

vr_bound = _get_interpolation_bounds(mesh_a.vr, v_scale * mesh_b.vr, "Vra", "Vrb")
vx_bound = _get_interpolation_bounds(mesh_a.vx, v_scale * mesh_b.vx, "Vxa", "Vxb")
x_bound = _get_interpolation_bounds(mesh_a.x, mesh_b.x, "Xa", "Xb")

vr_min = np.maximum(v_scale*vdiff_b.vr_left_bound[:, np.newaxis, np.newaxis, np.newaxis],
                    vdiff_a.vr_left_bound[np.newaxis, np.newaxis, :, np.newaxis])
vr_max = np.minimum(v_scale*vdiff_b.vr_right_bound[:, np.newaxis, np.newaxis, np.newaxis],
                    vdiff_a.vr_right_bound[np.newaxis, np.newaxis, :, np.newaxis])
vx_min = np.maximum(v_scale*vdiff_b.vx_left_bound[np.newaxis, :, np.newaxis, np.newaxis],
                    vdiff_a.vx_left_bound[np.newaxis, np.newaxis, np.newaxis, :])
vx_max = np.minimum(v_scale*vdiff_b.vx_right_bound[np.newaxis, :, np.newaxis, np.newaxis],
                    vdiff_a.vx_right_bound[np.newaxis, np.newaxis, np.newaxis, :])
condition = (vr_max > vr_min) & (vx_max > vx_min)
weight_value = 2*np.pi*(vr_max**2 - vr_min**2)*(vx_max - vx_min) / (
    vdiff_b.dvr_vol[:, np.newaxis, np.newaxis, np.newaxis] *
    vdiff_b.dvx[np.newaxis, :, np.newaxis, np.newaxis]
)
weight = np.where(condition, weight_value, 0)
weight = np.reshape(weight, (mesh_b.vr.size*mesh_b.vx.size, mesh_a.vr.size*mesh_a.vx.size), order='F')
fa_reshaped = np.reshape(fa, (mesh_a.vr.size*mesh_a.vx.size, mesh_a.x.size), order='F')
fb_on_xa = np.matmul(weight, fa_reshaped)

density = np.zeros(mesh_a.x.size)
vx_moment = np.zeros(mesh_a.x.size)
energy_moment = np.zeros(mesh_a.x.size)
for k in range(mesh_a.x.size):
    density[k] = np.sum(vdiff_a.dvr_vol * np.matmul(fa[:,:,k], vdiff_a.dvx))
    if density[k] > 0:
        vx_moment[k] = np.sqrt(mesh_a.Tnorm) * np.sum(vdiff_a.dvr_vol * np.matmul(fa[:,:,k], mesh_a.vx * vdiff_a.dvx)) / density[k]
        energy_moment[k] = mesh_a.Tnorm * np.sum(vdiff_a.dvr_vol * np.matmul(vdiff_a.vmag_squared * fa[:,:,k], vdiff_a.dvx)) / density[k]

fb = interp_fvrvxx(fa, mesh_a, mesh_b, do_warn=None, debug=False, correct=1)
print(json.dumps({
    "bounds": {
        "vr": [int(vr_bound.start), int(vr_bound.end)],
        "vx": [int(vx_bound.start), int(vx_bound.end)],
        "x": [int(x_bound.start), int(x_bound.end)]
    },
    "weight": weight.tolist(),
    "fa_reshaped": fa_reshaped.tolist(),
    "fb_on_xa": fb_on_xa.tolist(),
    "density": density.tolist(),
    "vx_moment": vx_moment.tolist(),
    "energy_moment": energy_moment.tolist(),
    "fb": fb.tolist(),
    "shape": list(fb.shape)
}))
"""

@testset "interp_fvrxx Python parity" begin
    if !_interp_python_available()
        @test_skip "interp_fvrxx Python parity requires python with numpy and scipy"
    else
        py = _interp_python_json(INTERP_PY_STAGE_SCRIPT)
        fa, mesh_a, mesh_b = _interp_fixture()
        vscale = sqrt(mesh_b.Tnorm / mesh_a.Tnorm)
        vdiff_a = VSpaceDifferentials(mesh_a.vr, mesh_a.vx)
        vdiff_b = VSpaceDifferentials(mesh_b.vr, mesh_b.vx)

        @testset "bounds and weights" begin
            vr_bound = KN1DJl._get_interpolation_bounds(mesh_a.vr, vscale .* mesh_b.vr)
            vx_bound = KN1DJl._get_interpolation_bounds(mesh_a.vx, vscale .* mesh_b.vx)
            x_bound = KN1DJl._get_interpolation_bounds(mesh_a.x, mesh_b.x)
            @test [vr_bound.start - 1, vr_bound.stop - 1] == Int.(py["bounds"]["vr"])
            @test [vx_bound.start - 1, vx_bound.stop - 1] == Int.(py["bounds"]["vx"])
            @test [x_bound.start - 1, x_bound.stop - 1] == Int.(py["bounds"]["x"])

            weight = KN1DJl._interp_fvrxx_weight_matrix(mesh_a, mesh_b, vdiff_a, vdiff_b, vscale)
            @test weight ≈ _fmat(py["weight"]) rtol=1e-12 atol=1e-14
        end

        @testset "Fortran flattening and fb_on_xa" begin
            weight = KN1DJl._interp_fvrxx_weight_matrix(mesh_a, mesh_b, vdiff_a, vdiff_b, vscale)
            fa_reshaped = reshape(fa, :, size(fa, 3))
            @test fa_reshaped ≈ _fmat(py["fa_reshaped"]) rtol=0 atol=0
            @test weight * fa_reshaped ≈ _fmat(py["fb_on_xa"]) rtol=1e-12 atol=1e-14
        end

        @testset "moments" begin
            density, vx_moment, energy_moment = KN1DJl._interp_fvrxx_moments(fa, mesh_a, vdiff_a)
            @test density ≈ _fvec(py["density"]) rtol=1e-12 atol=1e-14
            @test vx_moment ≈ _fvec(py["vx_moment"]) rtol=1e-12 atol=1e-14
            @test energy_moment ≈ _fvec(py["energy_moment"]) rtol=1e-12 atol=1e-14
        end

        @testset "full output and density rescale" begin
            fb_julia = interp_fvrxx(fa, mesh_a, mesh_b; correct=1)
            fb_python = _farray3(py["fb"])
            @test collect(size(fb_julia)) == Int.(py["shape"])
            @test maximum(abs.(fb_julia .- fb_python)) < 1e-10
            denom = max.(abs.(fb_python), 1e-12)
            @test maximum(abs.(fb_julia .- fb_python) ./ denom) < 1e-8
            @test isapprox(fb_julia, fb_python; rtol=1e-8, atol=1e-10)

            _, xmom_a, emom_a = KN1DJl._interp_fvrxx_moments(fa, mesh_a, vdiff_a)
            density_b, _, _ = KN1DJl._interp_fvrxx_moments(fb_julia, mesh_b, vdiff_b)
            @test all(isfinite, density_b)
            @test all(isfinite, xmom_a)
            @test all(isfinite, emom_a)
        end
    end
end
