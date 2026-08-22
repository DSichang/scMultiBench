The SPIRAL support package exactly as the benchmark ran it (spiral/: layers,
main, utils, CoordAlignment, model). It differs from tools_scripts/SPIRAL/spiral
in stability fixes only (single-thread BLAS/torch caps, DataLoader workers, a
sort flag); engine/drivers/run_spiral.py puts this directory first on sys.path
so the verified code is what runs. Process/ is taken from the upstream
tools_scripts/SPIRAL directory (identical code; only notebooks differ there).
