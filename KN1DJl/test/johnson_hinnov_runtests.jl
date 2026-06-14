using Test
using JSON
using Dierckx
import KN1DJl


const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PY_RTOL = 5e-8
const PY_ATOL = 0.0

const NE_REF = [1.0e17, 2.0e18, 5.0e20, 1.0e22]
const TE_REF = [0.5, 2.0, 20.0, 500.0]
const N0_REF = [1.0e15, 2.0e16, 3.0e17, 4.0e18]

function python_reference()::Dict{String,Any}
    script = raw"""
import json
import numpy as np

from KN1DPy.rates.johnson_hinnov.johnson_hinnov import Johnson_Hinnov

jh = Johnson_Hinnov()
ne = np.array([1.0e17, 2.0e18, 5.0e20, 1.0e22], dtype=float)
te = np.array([0.5, 2.0, 20.0, 500.0], dtype=float)
n0 = np.array([1.0e15, 2.0e16, 3.0e17, 4.0e18], dtype=float)

out = {
    "dknot": jh.dknot.tolist(),
    "tknot": jh.tknot.tolist(),
    "order": int(jh.order),
    "logr_shape": list(jh.logr_bscoef.shape),
    "logs_shape": list(jh.logs_bscoef.shape),
    "logalpha_shape": list(jh.logalpha_bscoef.shape),
    "jhr": {},
    "jhs": jh.jhs_coef(ne, te, no_null=True).tolist(),
    "jhalpha": jh.jhalpha_coef(ne, te, no_null=True).tolist(),
    "nh_saha": {},
    "lyman_alpha": jh.lyman_alpha(ne, te, n0, no_null=True).tolist(),
    "balmer_alpha": jh.balmer_alpha(ne, te, n0, no_null=True).tolist(),
}

for ion in (0, 1):
    for p in range(2, 7):
        out["jhr"][f"ion{ion}_p{p}"] = jh.jhr_coef(ne, te, ion, p, no_null=True).tolist()

for p in (1, 2, 3, 6):
    out["nh_saha"][f"p{p}"] = jh.nh_saha(ne, te, p).tolist()

print(json.dumps(out))
"""

    try
        return JSON.parse(read(Cmd(Cmd(["python3", "-c", script]); dir=REPO_ROOT), String))
    catch err
        error("Python parity tests require repo-local KN1DPy plus numpy/scipy importable by python3. Original error: $(err)")
    end
end

floatvec(x)::Vector{Float64} = Float64.(x)

function assert_py_match(actual::AbstractVector{<:Real}, expected; rtol::Float64=PY_RTOL, atol::Float64=PY_ATOL)
    expected_vec = floatvec(expected)
    @test length(actual) == length(expected_vec)
    # Python/SciPy and Julia/Dierckx both use FITPACK-style splines, but the wrappers
    # and Float32 coefficient loading differ slightly. A small relative tolerance
    # locks down numerical parity without baking in last-bit implementation noise.
    @test isapprox(Float64.(actual), expected_vec; rtol=rtol, atol=atol)
end

const PY_REF = python_reference()

@testset "Johnson-Hinnov Constructor And Data Invariants" begin
    jh = KN1DJl.JohnsonHinnov()

    @test jh.order == PY_REF["order"] == 4
    @test length(jh.dknot) == length(PY_REF["dknot"]) == 11
    @test length(jh.tknot) == length(PY_REF["tknot"]) == 15
    @test size(jh.logr_bscoef) == Tuple(Int.(PY_REF["logr_shape"])) == (5, 2, 77)
    @test size(jh.logs_bscoef) == Tuple(Int.(PY_REF["logs_shape"])) == (77,)
    @test size(jh.logalpha_bscoef) == Tuple(Int.(PY_REF["logalpha_shape"])) == (77,)
    @test length(jh.a_lyman) == 15
    @test length(jh.a_balmer) == 15
    @test issorted(jh.dknot)
    @test issorted(jh.tknot)
    @test size(jh.r_splines) == (2, 5)
    @test jh.s_spline isa Dierckx.Spline2D
    @test jh.alpha_spline isa Dierckx.Spline2D
    @test all(s -> s isa Dierckx.Spline2D, jh.r_splines)
    @test_throws ArgumentError KN1DJl.JohnsonHinnov(create=true)
end

@testset "Johnson-Hinnov Spline Helper Regression" begin
    jh = KN1DJl.JohnsonHinnov()
    lne = log.(NE_REF)
    lte = log.(TE_REF)

    spl = KN1DJl.jhr_spline(jh, 0, 2)
    @test spl isa Dierckx.Spline2D
    @test_throws ArgumentError KN1DJl.jhr_spline(jh, -1, 2)
    @test_throws ArgumentError KN1DJl.jhr_spline(jh, 0, 7)

    # Regression for the intended axis convention:
    # Python uses self.logr_bscoef.T[:, Ion, p-2], which maps to
    # Julia logr_bscoef[p-1, ion+1, :] and paired Dierckx.evaluate(lne, lte).
    direct = exp.(KN1DJl.bs2dr_jh(spl, lne, lte))
    assert_py_match(direct, PY_REF["jhr"]["ion0_p2"])
    @test direct == KN1DJl.jhr_coef(jh, NE_REF, TE_REF, 0, 2; no_null=true)

    @test_throws DimensionMismatch KN1DJl.bs2dr_jh(spl, [lne[1]], lte)
end

@testset "Johnson-Hinnov Coefficient Python Parity" begin
    jh = KN1DJl.JohnsonHinnov()

    for ion in (0, 1), p in 2:6
        assert_py_match(
            KN1DJl.jhr_coef(jh, NE_REF, TE_REF, ion, p; no_null=true),
            PY_REF["jhr"]["ion$(ion)_p$(p)"],
        )
    end

    assert_py_match(KN1DJl.jhs_coef(jh, NE_REF, TE_REF; no_null=true), PY_REF["jhs"])
    assert_py_match(KN1DJl.jhalpha_coef(jh, NE_REF, TE_REF; no_null=true), PY_REF["jhalpha"])

    for p in (1, 2, 3, 6)
        assert_py_match(KN1DJl.nh_saha(jh, NE_REF, TE_REF, p), PY_REF["nh_saha"]["p$(p)"])
    end

    assert_py_match(KN1DJl.lyman_alpha(jh, NE_REF, TE_REF, N0_REF; no_null=true), PY_REF["lyman_alpha"])
    assert_py_match(KN1DJl.balmer_alpha(jh, NE_REF, TE_REF, N0_REF; no_null=true), PY_REF["balmer_alpha"])
end

@testset "Johnson-Hinnov Out-Of-Range And Clamping Behavior" begin
    jh = KN1DJl.JohnsonHinnov()

    below_ne = exp(minimum(jh.dknot)) / 10.0
    above_ne = exp(maximum(jh.dknot)) * 10.0
    below_te = exp(minimum(jh.tknot)) / 10.0
    above_te = exp(maximum(jh.tknot)) * 10.0
    ne = [below_ne, above_ne, NE_REF[2]]
    te = [TE_REF[2], TE_REF[2], above_te]

    @test KN1DJl.jhr_coef(jh, ne, te, 0, 2; no_null=false) == fill(1.0e32, 3)
    @test KN1DJl.jhs_coef(jh, ne, te; no_null=false) == fill(1.0e32, 3)
    @test KN1DJl.jhalpha_coef(jh, ne, te; no_null=false) == fill(1.0e32, 3)

    clamped_ne = [exp(minimum(jh.dknot)), exp(maximum(jh.dknot)), NE_REF[2]]
    clamped_te = [TE_REF[2], TE_REF[2], exp(maximum(jh.tknot))]

    @test KN1DJl.jhr_coef(jh, ne, te, 0, 2; no_null=true) ≈
          KN1DJl.jhr_coef(jh, clamped_ne, clamped_te, 0, 2; no_null=false) rtol=PY_RTOL
    @test KN1DJl.jhs_coef(jh, ne, te; no_null=true) ≈
          KN1DJl.jhs_coef(jh, clamped_ne, clamped_te; no_null=false) rtol=PY_RTOL
    @test KN1DJl.jhalpha_coef(jh, ne, te; no_null=true) ≈
          KN1DJl.jhalpha_coef(jh, clamped_ne, clamped_te; no_null=false) rtol=PY_RTOL
end

@testset "Johnson-Hinnov Invalid Inputs And Edge Cases" begin
    jh = KN1DJl.JohnsonHinnov()

    @test KN1DJl.jhr_coef(jh, Float64[], Float64[], 0, 2) == Float64[]
    @test KN1DJl.jhs_coef(jh, Float64[], Float64[]) == Float64[]
    @test KN1DJl.jhalpha_coef(jh, Float64[], Float64[]) == Float64[]
    @test KN1DJl.nh_saha(jh, Float64[], Float64[], 1) == Float64[]
    @test KN1DJl.lyman_alpha(jh, Float64[], Float64[], Float64[]) == Float64[]
    @test KN1DJl.balmer_alpha(jh, Float64[], Float64[], Float64[]) == Float64[]

    @test_throws DimensionMismatch KN1DJl.jhr_coef(jh, [1.0, 2.0], [1.0], 0, 2)
    @test_throws DimensionMismatch KN1DJl.jhs_coef(jh, [1.0, 2.0], [1.0])
    @test_throws DimensionMismatch KN1DJl.jhalpha_coef(jh, [1.0, 2.0], [1.0])
    @test_throws DimensionMismatch KN1DJl.nh_saha(jh, [1.0, 2.0], [1.0], 1)
    @test_throws DimensionMismatch KN1DJl.lyman_alpha(jh, [1.0, 2.0], [1.0], [1.0, 2.0])
    @test_throws DimensionMismatch KN1DJl.lyman_alpha(jh, [1.0, 2.0], [1.0, 2.0], [1.0])
    @test_throws DimensionMismatch KN1DJl.balmer_alpha(jh, [1.0, 2.0], [1.0], [1.0, 2.0])
    @test_throws DimensionMismatch KN1DJl.balmer_alpha(jh, [1.0, 2.0], [1.0, 2.0], [1.0])

    @test_throws ArgumentError KN1DJl.jhr_coef(jh, NE_REF, TE_REF, -1, 2)
    @test_throws ArgumentError KN1DJl.jhr_coef(jh, NE_REF, TE_REF, 2, 2)
    @test_throws ArgumentError KN1DJl.jhr_coef(jh, NE_REF, TE_REF, 0, 1)
    @test_throws ArgumentError KN1DJl.jhr_coef(jh, NE_REF, TE_REF, 0, 7)
    @test_throws ArgumentError KN1DJl.nh_saha(jh, NE_REF, TE_REF, 0)

    @test KN1DJl.nh_saha(jh, [-1.0, 1.0e17, 1.0e33], [1.0, -1.0, 1.0], 1) == fill(1.0e32, 3)
end
