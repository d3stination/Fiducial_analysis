# Dataset setup

Raw recordings are external inputs and are not part of this repository.

Set `P1_DATA_ROOT` to a local folder with the following structure:

```text
Variability Analysis/
  Heart Cycle/
    *.h5
  IITB ICG/
    Normal Data/
      *.mat
    Excerise/
      *.mat
  ReBeatICG/
    Raw/
      Baseline/
        *.mat
      Cognetive Workload/
        *.mat
    Annotations/
      Baseline/
      Cognetive Workload/
```

The folder spellings `Excerise` and `Cognetive Workload` reproduce the source layout and are not corrected by the pipeline.

The full upstream project also stores generated Phase 1 to Phase 4 files under `${P1_DATA_ROOT}/outputs/`. The HeartCycle replay in this repository rebuilds those stages in a temporary release-local folder and compares them with the external canonical files. It never overwrites the external data root.

HeartCycle reference alignment requires the ECHO event channels in the HDF5 files. ReBeat expert annotation files are separate from raw recordings. IITB Doppler measurements are not treated as synchronized B/X annotations.

Copy `configs/paths.example.yaml` to `configs/paths.local.yaml` for local notes. The executable replay uses the `--data-root` argument so commands do not depend on a private absolute path committed to Git.
