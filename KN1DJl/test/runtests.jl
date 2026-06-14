using Test
using KN1DJl

include("janev_runtests.jl")
include("johnson_hinnov_runtests.jl")
include("python_compare_runtests.jl")
include("kinetic_h_runtests.jl")
include("kinetic_h2_runtests.jl")
include("kn1d_lite_runtests.jl")
include("diiid_runtests.jl")
include("interp_fvrxx_runtests.jl")
    
CONFIG_PATH = joinpath(@__DIR__, "..", "config.json")

@testset "Config Parsing" begin
    cfg = @inferred get_config(CONFIG_PATH)
    @test cfg isa KN1DConfig
    @test cfg.kinetic_h.mesh_size == 10
    @test cfg.kinetic_h.ion_rate == "adas"
    @test cfg.kinetic_h2.mesh_size == 6
    @test cfg.collisions.SIMPLE_CX
end

@testset "Interpolation Inference" begin
    x = [0.0, 1.0, 2.0]
    y = [0.0, 10.0, 20.0]

    itp_const = @inferred linear_interp_1d(x, y)
    itp_extrap = @inferred linear_interp_1d(x, y; fill_value="extrapolate")
    scalar_interp = @inferred interp_1d(x, y, 1.5)
    vector_interp = @inferred interp_1d(x, y, [0.5, 1.5])
    extrap_interp = @inferred interp_1d(x, y, [-1.0, 3.0]; fill_value="extrapolate")
    out = Vector{Float64}(undef, 2)
    outside_const = @inferred itp_const(3.0)

    @test @inferred(itp_const(1.5)) == 15.0
    @test isnan(outside_const)
    @test @inferred(itp_extrap(3.0)) == 30.0
    @test @inferred(evaluate!(out, itp_extrap, [0.5, 1.5])) == [5.0, 15.0]

    @test scalar_interp == 15.0
    @test vector_interp == [5.0, 15.0]
    @test extrap_interp == [-10.0, 30.0]
    @test_throws ArgumentError linear_interp_1d([1.0, 0.0], y; assume_sorted=true)
end

@testset "Utility Inference" begin
    @test @inferred(poly(2.0, [1.0, 2.0, 3.0])) == 17.0
    @test @inferred(poly([0.0, 1.0], [1.0, 2.0])) == [1.0, 3.0]
    @test @inferred(kn1d_lite_output_placeholder()) isa KN1DLiteOutput
end
