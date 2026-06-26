uv run python -m toy_3d_pointcloud_et scsi --channel cryoet --n-tilts 11 --tilt-step 12 --tilt-axis y --shape torus cylinder \ # base object(s)

--dataset template --dataset-eps 0.1 \ # N perturbed copies, ‖δ‖ ≤ 0.1

--n-objects 128 \ # N

--bootstrap tomo