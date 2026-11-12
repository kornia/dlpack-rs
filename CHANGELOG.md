# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.4.0 - 2026-11-12

### Changed

- Breaking: the `pyo3` glue now targets pyo3 0.30 (drops Python 3.8 and 3.13t with it).
  No API changes to `ffi` or `safe`.

## 0.3.4 - 2026-07-06

### Changed

- Vendor dlpack as the dmlc/dlpack git submodule and regenerate the bindings (#9).

## 0.3.3 - 2026-07-06

### Changed

- Inline dlpack-sys into dlpack-rs; publish a single crate (#8).
