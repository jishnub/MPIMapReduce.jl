using Test
using MPIMapReduce
using MPI

const comm = MPI.COMM_WORLD
const np = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

function reductiontest(op)
	arraysize = (10, 1)
	for n in 1:5, (fmap, itrs) in Any[
		(x -> ones(arraysize) * x, (1:n,)),
		((x, y) -> ones(arraysize) * (x + y), (1:n, 1:n)),
		]

		yexp = mapreduce(fmap, op, itrs...)

		for root in 0:MPI.Comm_size(comm) - 1
			y_Julia = MPIMapReduce.pmapreduce(fmap, op, itrs..., root = root)
			if MPI.Comm_rank(comm) == root
				@test y_Julia == yexp
			else
				@test y_Julia == nothing
			end
		end
	end

	@test_throws ArgumentError MPIMapReduce.pmapreduce(x -> ones(2), op, 1:0)
	@test_throws ArgumentError MPIMapReduce.pmapreduce(x -> ones(2), op, 1:1, root = -1)
	@test_throws ArgumentError MPIMapReduce.pmapreduce(x -> ones(2), op, 1:1, root = np + 1)
end

@testset "Reduction" begin
	@testset "elementwise sum" begin
		reductiontest(+)
	end
	@testset "elementwise product" begin
		reductiontest((x,y) -> x .* y)
	end
	@testset "elementwise max" begin
		reductiontest((x,y) -> max.(x,y))
	end
	@testset "elementwise min" begin
		reductiontest((x,y) -> min.(x,y))
	end
end

function concattest(op)
	for n in 1:10, (fmap, itrs) in  Any[
		(x -> x^2, (1:n,)),
		((x, y) -> (x^2 + y^2), (1:n, 1:n)),
		(x -> ones(2) * x^2, (1:n,)),
		((x, y) -> ones(2) * (x^2 + y^2), (1:n, 1:n)),
		(x -> ones(1,2) * x^2, (1:n,)),
		((x, y) -> ones(1,2) * (x^2 + y^2), (1:n, 1:n)),
		(x -> ones(2,2,2) * x^2, (1:n,)),
		((x, y) -> ones(2,2,2) * (x^2 + y^2), (1:n, 1:n)),
		]

		yexp = mapreduce(fmap, op, itrs...)

		for root in 0:MPI.Comm_size(comm) - 1
			y = MPIMapReduce.pmapreduce(fmap, op, itrs..., root = root)
			if MPI.Comm_rank(comm) == root
				@test begin
					res = y == yexp
					if !res
						@show n, np, root, y, yexp
					end
					res
				end
			else
				@test y === nothing
			end
		end
	end

	@test_throws ArgumentError MPIMapReduce.pmapreduce(x -> ones(2), op, 1:0)
	@test_throws ArgumentError MPIMapReduce.pmapreduce(x -> ones(2), op, 1:1, root = -1)
	@test_throws ArgumentError MPIMapReduce.pmapreduce(x -> ones(2), op, 1:1, root = np + 1)
end

@testset "Concatenation" begin
	@testset "vcat" begin
		rank == 0 && @info "vcat"
	    concattest(vcat)
	end

	@testset "hcat" begin
		rank == 0 && @info "hcat"
		concattest(hcat)
	end

	@testset "Cat" begin
	    @testset "single dimension" begin
	    	@testset "Cat(1) is vcat" begin
	    		rank == 0 && @info "Cat(1) == vcat"
	    		fmap = x -> x^2
	    		for itrs in [(1:1,), (1:4,)]
	    			yexp = mapreduce(fmap, vcat, itrs...)
	    			for root in 1:MPI.Comm_size(comm) - 1
			    	    y1 = MPIMapReduce.pmapreduce(fmap, MPIMapReduce.Cat(1), itrs..., root = root)
			    	    y2 = MPIMapReduce.pmapreduce(fmap, vcat, itrs..., root = root)
			    	    if MPI.Comm_rank(comm) == root
					    	@test y1 == y2 == yexp
					    else
					    	@test y1 === y2 === nothing
					    end
			    	end
		    	end
	    	end
	    	@testset "Cat(2) is hcat" begin
	    		rank == 0 && @info "Cat(2) == hcat"
	    		fmap = x -> x^2
	    		for itrs in [(1:1,), (1:4,)]
	    			yexp = mapreduce(fmap, hcat, itrs...)
	    			for root in 1:MPI.Comm_size(comm) - 1
					    y1 = MPIMapReduce.pmapreduce(fmap, MPIMapReduce.Cat(2), itrs..., root = root)
					    y2 = MPIMapReduce.pmapreduce(fmap, hcat, itrs..., root = root)
					    if MPI.Comm_rank(comm) == root
					    	@test y1 == y2 == yexp
					    else
					    	@test y1 === y2 === nothing
					    end
					end
				end
	    	end
	    	rank == 0 && @info "Cat single dimension"
	        concattest(MPIMapReduce.Cat(3))
	        concattest(MPIMapReduce.Cat(Val(3)))
	        concattest(MPIMapReduce.Cat(4))
	        concattest(MPIMapReduce.Cat(Val(4)))
	    end
	    @testset "multiple dimensions" begin
	    	rank == 0 && @info "Cat multiple dimensions"
	        concattest(MPIMapReduce.Cat(3:4))
	        concattest(MPIMapReduce.Cat([3,4]))
	        concattest(MPIMapReduce.Cat([1,4]))
	    end
	end
end
