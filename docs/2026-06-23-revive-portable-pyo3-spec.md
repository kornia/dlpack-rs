# Revive dlpack-rs — portable FFI + safe wrappers + pyo3 glue

**Date:** 2026-06-23
**Repo:** github.com/kornia/dlpack-rs · branch `feat/revive-portable-pyo3`
**Consumer:** kornia-rs `kornia-py` (task #24: numpy-agnostic Image + `__dlpack__` for Torch/CuPy/TensorRT on Jetson Orin aarch64).

## Problem

The crate (`dlpack-rs 0.1.2`) is an abandoned bindgen dump: `build.rs` runs `bindgen` on a `dlpack` git submodule and commits an **x86-hardcoded** `dlpack.rs` (`__GLIBC__`, `__WORDSIZE`). It fails to build from crates.io (submodule absent in the published artifact) and won't compile on aarch64. No safe wrappers, no pyo3 glue.

## Goal

A portable, dependency-light DLPack crate with hand-written FFI, ergonomic safe wrappers, and optional pyo3 capsule glue — usable on aarch64 (and anywhere), suitable as the DLPack backbone for kornia-py.

## Decisions (locked with user 2026-06-23)

1. **One crate, pyo3 behind a feature.** `ffi` + `safe` always compile (no pyo3 dep); the `__dlpack__`/capsule glue is gated behind a `pyo3` cargo feature. `default = []`.
2. **Git-tag dependency first.** Tag `v0.2.0`; kornia-py depends via `{ git, tag = "v0.2.0" }`. Publish to crates.io later once it settles.
3. Hand-written portable FFI — **remove `bindgen`, `build.rs`, the `dlpack` submodule, and the committed `dlpack.rs`**.
4. Support both legacy `DLManagedTensor` and DLPack-1.0 `DLManagedTensorVersioned`.
5. CPU-first; CUDA device/stream are out of scope for this revival (leave the device enum + constructors so it's a later wiring, not a redesign).

## Architecture

```
src/
  lib.rs        re-exports; crate docs; #![no_std]-friendly core (std only where needed)
  ffi.rs        #[repr(C)] ABI structs — always compiled, no deps
  safe.rs       TensorInfo builder, dtype/device constructors, ManagedTensor ownership
  pyo3_glue.rs  IntoDLPack + PyTensor — #[cfg(feature = "pyo3")]
```

### `ffi.rs` — ABI structs (match dmlc/dlpack `dlpack.h`)
- `DLDeviceType` constants (`kDLCPU = 1`, `kDLCUDA = 2`, `kDLCUDAHost = 3`, … `kDLOneAPI = 14`, `kDLWebGPU = 15`).
- `DLDevice { device_type: i32, device_id: i32 }` (enum is C `int`).
- `DLDataTypeCode` constants (`kDLInt=0, kDLUInt=1, kDLFloat=2, kDLOpaqueHandle=3, kDLBfloat=4, kDLComplex=5, kDLBool=6`).
- `DLDataType { code: u8, bits: u8, lanes: u16 }`.
- `DLTensor { data: *mut c_void, device: DLDevice, ndim: i32, dtype: DLDataType, shape: *mut i64, strides: *mut i64, byte_offset: u64 }`.
- `DLManagedTensor { dl_tensor: DLTensor, manager_ctx: *mut c_void, deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)> }`.
- `DLPackVersion { major: u32, minor: u32 }`; flag consts `DLPACK_FLAG_BITMASK_READ_ONLY = 1`, `DLPACK_FLAG_BITMASK_IS_COPIED = 2`.
- `DLManagedTensorVersioned { version: DLPackVersion, manager_ctx: *mut c_void, deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>, flags: u64, dl_tensor: DLTensor }`.
All `#[repr(C)]`, `Copy` where layout-only. No `bindgen`, no arch cfg.

### `safe.rs` — ownership-safe builders
- `TensorInfo { data: *mut c_void, device: DLDevice, dtype: DLDataType, shape: Vec<i64>, strides: Option<Vec<i64>>, byte_offset: u64 }` with `contiguous(...)`, `strided(...)`, `with_byte_offset(...)`.
- dtype ctors: `dtype_u8/u16/u32/i8/i16/i32/i64/f16/f32/f64/bool()`; device ctors `cpu_device()`, `cuda_device(id)`.
- **`ManagedContext<T>`**: heap box owning `(T /*keep-alive*/, shape: Vec<i64>, strides: Option<Vec<i64>>)`, with the `DLManagedTensor[Versioned]` embedded so `shape`/`strides` pointers stay valid for the tensor's lifetime. A `deleter` `extern "C" fn` reconstructs the `Box<ManagedContext<T>>` from `manager_ctx` and drops it (frees buffer + shape). This is the crux: the produced tensor keeps its backing buffer alive until the consumer calls the deleter.
- `pack<T>(keepalive: T, info: TensorInfo) -> *mut DLManagedTensor` and `pack_versioned<T>(..., flags) -> *mut DLManagedTensorVersioned`.

### `pyo3_glue.rs` (feature `pyo3`)
- `trait IntoDLPack: Sized { fn tensor_info(&self) -> TensorInfo; fn into_capsule(self, py) -> PyResult<Py<PyAny>>; fn into_capsule_versioned(self, py, read_only: bool) -> PyResult<Py<PyAny>>; }` — produces a `PyCapsule` named `"dltensor"` (legacy) / `"dltensor_versioned"` (1.0). Default impls call `safe::pack`.
- Capsule **destructor**: if the capsule is still named `"dltensor"`/`"dltensor_versioned"` when GC'd (consumer never took it), call the managed tensor's own `deleter` to avoid a leak; if renamed to `"used_dltensor"`, do nothing (consumer owns it).
- `struct PyTensor { capsule: Py<PyCapsule>, mt: *mut DLManagedTensorVersioned-or-legacy }` — `from_pyany(py, obj)` calls `obj.__dlpack__()`, validates the capsule name, **renames it to `"used_dltensor"`** (double-free guard), and exposes `shape()/strides()/dtype()/device()/data_ptr()/byte_offset()/is_contiguous()/numel()/nbytes()/is_read_only()`. `Drop` calls the producer's `deleter`. `PyTensor` holds the capsule alive; any borrow of its data must not outlive it.

## Testing

- **ffi**: `size_of`/`align_of` assertions for every struct against the documented ABI (catch field-order/padding mistakes); device/dtype constant values.
- **safe**: build a `TensorInfo` from a `Vec<f32>`, `pack` it, manually invoke the `deleter`, assert no leak/UAF (miri-style reasoning + a drop-counter test using a keep-alive type whose `Drop` flips a flag).
- **pyo3** (feature `pyo3`): compile gate; a capsule-name/rename unit test where feasible without a Python interpreter, plus a `#[test]` behind `Python::with_gil` if the dev env has libpython (else doc/ignored). Full Torch round-trip lives in kornia-py (task #24).
- CI builds **both** `--no-default-features` and `--features pyo3`, on `ubuntu-latest` AND an aarch64 job.

## Deliverables

- `ffi.rs`, `safe.rs`, `pyo3_glue.rs`, rewritten `lib.rs`.
- `Cargo.toml`: drop `bindgen`/`build.rs`; `[features] pyo3 = ["dep:pyo3"]`; `pyo3 = { version = "0.28", optional = true }`; version `0.2.0`; keep `license = "Apache-2.0"`; edition 2021.
- Remove `build.rs`, `dlpack.rs`, `.gitmodules` + `dlpack` submodule.
- Add `LICENSE` (Apache-2.0). Rewrite `README.md` (usage: FFI, safe, pyo3 `__dlpack__`). Rewrite `.github/workflows/ci.yml` (no submodule/bindgen/auto-commit/auto-publish; matrix default+pyo3, x86+aarch64; clippy + test).
- Tag `v0.2.0`.

## Out of scope (future)
- CUDA device/stream sync; crates.io publish; the kornia-py `__dlpack__`/`from_dlpack` wiring (task #24) and the numpy-agnostic Image core.
