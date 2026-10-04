# Preserved RQ3 evidence

These verified working copies are included in the RQ3 branch. Analysis reads the repository copies; no external download directory is required. See the [RQ3 output guide](../README.md) for results, image browsing and integrity checks.

- calibration/: corrected calibration bundle, images, both response passes, sessions and metadata-correction audit. Duplicate working copies are excluded from Git. The calibration audit and repeatability files are also available in reports/rq3_chatgpt_calibration/.
- chatgpt_main/: original blind judging bundle, all 100 response files, session log and the 200 validated scores. Original manifest/image hashes are preserved.
- analysis/: all 76 inventoried NSCC report/plot/record/provenance files plus TRANSFER.json. Includes all 800 original generation records and the 200 scores.
- transfer_checksums/: original archive/chunk/returned-judgment checksums and download verification reports. The transport ZIPs, chunk payloads and disposable Git clones are not needed for analysis.
- COPY_VERIFICATION.json: historical source-to-repository SHA-256 inventory and transfer commit. Original source locations are recorded for provenance; they are not runtime dependencies. The later wording change in calibration/CORRECTIONS.md is recorded separately in reports/rq3_shared_outputs.json, which provides current sharing checksums.

Run the commands in RQ3_experiment.md from the repository root. Derived charts and discussion are in outputs/rq3/postprocessing/. Original NSCC absolute paths inside evidence are intentionally unchanged; post-processing resolves images through the local bundle using a separate mapping.

The archive is sufficient for the completed quality/timing/region analysis. It is not a full backup of every NSCC timing-repeat image, pilot generation or checkpoint. Preserve those NSCC experiment records in durable storage before deleting them. Model weights, caches and duplicate transfer folders are not included in Git.
