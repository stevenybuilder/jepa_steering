# New Jersey paused-work archive

This is a byte-preserving split of `nj-paused-evidence.tar.gz`, not new results.
The archive contains 527 selected files, including the latest complete resume
checkpoint in each selected training directory. Older checkpoint histories stay
on the stopped source instance 50239185, whose storage must not be destroyed.
Partial behavioral streams are not completed experiments or resumable RNG state.

Download all twelve `nj-paused-evidence.tar.gz.part-0000` through `part-0011`
files, `nj-PARTS.json`, and `nj-FILES.json` from this private folder. `nj-PARTS.json`
records their exact order, Drive IDs, byte sizes, and SHA256 hashes. Join them in
that order to a new file named `nj-paused-evidence.tar.gz` (do not overwrite an
existing archive), checking each part against that manifest.

The rejoined archive must have exactly **1,176,313,887 bytes** and SHA256:

`eba8e3dd4fe48404e45508fa9e14b0be948ae4587eab7af480a4fe19aaffbc2e`

Before extracting, verify the complete compressed-byte hash and every tar member
against `nj-FILES.json`. The repository's `backup_results_to_google.verify_archive`
does both without extracting. Preserve the original provenance receipts; this
backup does not establish full-study or fresh-confirmation completion.

An independently verified unsplit fallback is available to the same Google
account at:

`gs://rgt-jepa-archive-2026/rep_geometry_transcoder/core-priority-pause-20260908-v1/nj-paused-evidence.tar.gz`

`nj-DRIVE_VERIFIED.json`, once present, is the full streamed Drive readback
receipt. Upload acknowledgements or per-part checks alone are not that receipt.
