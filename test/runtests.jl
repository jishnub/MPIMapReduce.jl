using MPI

juliapath = joinpath(Sys.BINDIR, "julia")

for np in 1:4
	@info "tests with np = $np"
	cmd = `$(MPI.mpiexec_path) -np $np $juliapath --startup=no --project tests.jl`
	run(cmd)
end
