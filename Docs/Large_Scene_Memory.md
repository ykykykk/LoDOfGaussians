# Large scenes on a single GPU

## Recommended execution plan

1. Keep the active GPU working set within physical VRAM. Total node count and the
   number rendered in one view are separate budgets.
2. Use growing RAM backing while the working set fits physical memory. A large
   Windows pagefile provides commit capacity, not fast random-access memory.
3. Use explicit SSD backing only for models that exceed RAM. Place its directory
   and the frequently read images/checkpoints on NVMe. Retain final exports on the
   archive disk if desired. No dataset is moved by these settings.
4. Measure actual step time, cache transfers and selected LoD before increasing
   detail further. More points alone does not guarantee better reconstruction.

On this workstation, inspection on 2026-09-28 found a 24 GiB RTX 3090, 64 GiB RAM,
D: on a WD mechanical disk, and F: on a Samsung 980 PRO NVMe. The configured F:
pagefile was 20,000 MiB initial / 200,000 MiB maximum, currently allocated 20,000 MiB.
The user subsequently moved the DJI dataset and outputs to
`F:\Resources\Projects\3D\dji_test`, so its current working data is on NVMe.

At SH degree 1, CPU parameters and Adam moments use 276 bytes per total node;
hierarchy and scores add 28 bytes. Thus 90 million total nodes require about
25.5 GiB for these arrays alone, and 200 million require about 56.6 GiB. These
figures exclude hierarchy rebuilds, visibility metadata, images and GPU allocations.
The cap counts internal hierarchy nodes as well as exported leaves.

## Implemented controls

- RAM backing starts from the existing model plus two bounded split windows and
  grows by 25% when required. It preserves parameters, Adam moments and split scores.
  Unbounded split configurations retain full-capacity allocation.
- `resident.host_storage`: `ram` (default) or `mmap`. The latter requires
  `resident.host_directory`, preferably on NVMe. Each run creates a unique scratch
  directory. Fresh mapped files reserve capacity without explicitly zeroing every
  page. Scratch files are **not checkpoints** and are retained after exit; remove
  that run's directory only when its process has exited and a valid checkpoint exists.
  SSD mapping still uses the OS file cache and can be slow under heavy random I/O.
- `resident.max_active_nodes`: zero preserves the normal cut. A positive value
  chooses a coarser valid LoD cut when a view exceeds the budget. It does not discard
  arbitrary rows. This changes view detail and training coverage and is not a VRAM
  guarantee: pixel count and splat overlap also affect renderer memory.
- Checkpoints remain normal version-1 PyTorch files. Synchronous saving shares only
  the live prefix storage, eliminating a full-state clone and avoiding serialization
  of unused capacity. Do not save concurrently with parameter mutation.
- Loading warms large files sequentially when they fit within half the available
  RAM, then validates/copies bounded row chunks.
  Existing checkpoints remain readable. Warmup reads the whole file and may have
  little benefit on an already warm SSD.
- Profiles now include `host_capacity`, `host_storage`, `max_active_nodes` and
  `selected_lod_multiplier` so capacity and quality tradeoffs are visible.

## Prepared DJI profiles

`configs/dji_90m_ram.json` is the first choice for this machine. It sets a 90-million
total-node cap, 1.2-million new nodes per split window, and a 10-million active-node
budget. `configs/dji_90m_ssd.json` uses identical training settings with SSD backing
under `.resident_cache` in the project working directory (F: on this workstation).

Both retain 60,000 total fine steps and the 48,000-step densification cutoff. From
the 35,000 checkpoint there are 17 remaining split windows at interval 743: even if
all saturate, the run reaches approximately 43.6 million total nodes, not the
90-million ceiling. Reaching the ceiling requires a separately extended schedule.
Use `--resume_checkpoint` and `--allow_growth_resume` for the first migration from
the earlier 30-million configuration; subsequent resumes use the new run's checkpoint.

These profiles have not been benchmarked on the full DJI run. The memory and I/O
changes remove known unnecessary allocations; they do not establish a measured
speedup or prove that a 90-million-node model fits every full-resolution view.
Full-scene hierarchy rebuilds and GPU hierarchy metadata still scale with total
model size. This is not a fully spatially partitioned out-of-core training engine.

## Relocating existing training data

`python -m tools.relocate_resident_checkpoint --input OLD.pt --output NEW.pt
--old-root OLD_DIRECTORY --new-root NEW_DIRECTORY --manifest SCAFFOLD_MANIFEST`
verifies the existing metadata/calibration fingerprint using the old logical paths,
writes a new checkpoint with relocated source/hierarchy paths, and updates the
supplied scaffold manifest while preserving its previous version. It does not
disable normal resume contract validation or overwrite the original checkpoint.
Restore coarse-model junctions separately if moving directories left them empty.
