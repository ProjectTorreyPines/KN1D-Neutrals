using Test
import KN1DJl

@testset "Janev Scalar Inference" begin
    @test @inferred(KN1DJl.sigma_cx_h0(1.0)) isa Float64
    @test @inferred(KN1DJl.sigma_cx_h0(1.0; freeman=true)) isa Float64
    @test @inferred(KN1DJl.sigma_cx_hh(1.0)) isa Float64
    @test @inferred(KN1DJl.sigma_el_p_h(2.0)) isa Float64
    @test @inferred(KN1DJl.sigmav_cx_h0(2.0, 3.0)) isa Float64
    @test @inferred(KN1DJl.sigmav_cx_hh(2.0, 3.0)) isa Float64
    @test @inferred(KN1DJl.sigmav_ion_h0(5.0)) isa Float64
    @test @inferred(KN1DJl.sigmav_rec_h1s(5.0)) isa Float64
end

@testset "Janev Vector Shapes" begin
    x = [0.1, 1.0, 10.0]
    y = [1.0, 2.0, 3.0]

    @test length(KN1DJl.sigma_cx_h0(x)) == 3
    @test length(KN1DJl.sigma_cx_hh(x)) == 3
    @test length(KN1DJl.sigma_el_h_h(x)) == 3
    @test length(KN1DJl.sigmav_cx_h0(y, x)) == 3
    @test length(KN1DJl.sigmav_cx_hh(y, x)) == 3
    @test length(KN1DJl.sigmav_ion_hh(x)) == 3
    @test length(KN1DJl.sigmav_p_p_hp(x)) == 3
end

@testset "Janev Sanity" begin
    @test KN1DJl.sigma_cx_h0(1.0) > 0.0
    @test KN1DJl.sigma_el_h_h(1.0) > 0.0
    @test KN1DJl.sigmav_ion_h0(1.0) > 0.0
    @test KN1DJl.sigmav_rec_h1s(1.0) > 0.0
    @test_throws ArgumentError KN1DJl.sigmav_cx_h0([1.0, 2.0], [1.0])
end
