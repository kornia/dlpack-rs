# dlpack-rs revival — implementation plan

> Execute with superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Replace the bindgen/x86 dump with portable FFI + safe wrappers + optional pyo3 glue; aarch64-clean; tag v0.2.0.

**Repo/branch:** /home/nvidia/dlpack-rs · `feat/revive-portable-pyo3`. Spec: `docs/2026-06-23-revive-portable-pyo3-spec.md`.

## Global Constraints
- Hand-written FFI only — NO `bindgen`, NO `build.rs`, NO `dlpack` submodule, NO committed `dlpack.rs`.
- `default = []`; pyo3 strictly behind `[features] pyo3`. `ffi`/`safe` must compile with `--no-default-features`.
- `#[repr(C)]` on every FFI struct; field order/types EXACT per dmlc/dlpack `dlpack.h`.
- edition 2021, license Apache-2.0, version 0.2.0. Commit per task.
- Unsafe is confined to `ffi` (layout) and the deleter/pack code in `safe`/`pyo3_glue`; every `unsafe` block carries a `// SAFETY:` note.

---

### Task 1: Portable FFI + manifest cleanup + lib wiring

**Files:** create `src/ffi.rs`; rewrite `src/lib.rs`, `Cargo.toml`; delete `build.rs`, `dlpack.rs`, `.gitmodules`, the `dlpack/` submodule dir; add `LICENSE`.

**Produces:** `dlpack_rs::ffi::{DLDevice, DLDataType, DLTensor, DLManagedTensor, DLManagedTensorVersioned, DLPackVersion}` + device/dtype-code constants.

- [ ] **Step 1: Cargo.toml** — replace with:
```toml
[package]
name = "dlpack-rs"
description = "Portable Rust bindings for the DLPack protocol (FFI + safe wrappers + optional pyo3 glue)"
repository = "https://github.com/kornia/dlpack-rs"
version = "0.2.0"
license = "Apache-2.0"
edition = "2021"

[features]
default = []
pyo3 = ["dep:pyo3"]

[dependencies]
pyo3 = { version = "0.28", optional = true }
```
(No `[build-dependencies]`, no `build = ...`.)

- [ ] **Step 2: remove dead infra**
```
git rm -q build.rs dlpack.rs .gitmodules
git rm -qr dlpack 2>/dev/null || true
rm -rf .git/modules/dlpack 2>/dev/null || true
```

- [ ] **Step 3: add LICENSE** — write the standard Apache-2.0 license text to `LICENSE` (full text; `curl -s https://www.apache.org/licenses/LICENSE-2.0.txt` if available, else the canonical text).

- [ ] **Step 4: write `src/ffi.rs`** (ABI-exact; values from dmlc/dlpack):
```rust
//! Raw `#[repr(C)]` DLPack ABI types. No dependencies; portable across arches.
use core::ffi::c_void;

// DLDeviceType (C enum = int). Only the common kinds; extend as needed.
pub const K_DL_CPU: i32 = 1;
pub const K_DL_CUDA: i32 = 2;
pub const K_DL_CUDA_HOST: i32 = 3;
pub const K_DL_OPENCL: i32 = 4;
pub const K_DL_VULKAN: i32 = 7;
pub const K_DL_METAL: i32 = 8;
pub const K_DL_ROCM: i32 = 10;
pub const K_DL_CUDA_MANAGED: i32 = 13;
pub const K_DL_ONE_API: i32 = 14;

// DLDataTypeCode
pub const K_DL_INT: u8 = 0;
pub const K_DL_UINT: u8 = 1;
pub const K_DL_FLOAT: u8 = 2;
pub const K_DL_OPAQUE_HANDLE: u8 = 3;
pub const K_DL_BFLOAT: u8 = 4;
pub const K_DL_COMPLEX: u8 = 5;
pub const K_DL_BOOL: u8 = 6;

// DLPack 1.0 flags
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1 << 0;
pub const DLPACK_FLAG_BITMASK_IS_COPIED: u64 = 1 << 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDevice { pub device_type: i32, pub device_id: i32 }

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDataType { pub code: u8, pub bits: u8, pub lanes: u16 }

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64, // null => compact row-major
    pub byte_offset: u64,
}

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLPackVersion { pub major: u32, pub minor: u32 }

#[repr(C)]
pub struct DLManagedTensorVersioned {
    pub version: DLPackVersion,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    pub flags: u64,
    pub dl_tensor: DLTensor,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};
    #[test]
    fn abi_layout() {
        assert_eq!(size_of::<DLDevice>(), 8);
        assert_eq!(size_of::<DLDataType>(), 4);
        // DLTensor: ptr(8)+DLDevice(8)+ndim(4)+pad(4)+DLDataType(4)+pad(4)+ptr(8)+ptr(8)+u64(8)
        assert_eq!(size_of::<DLTensor>(), 56);
        assert_eq!(align_of::<DLTensor>(), 8);
        assert_eq!(size_of::<DLManagedTensor>(), 56 + 8 + 8);
        assert_eq!(size_of::<DLPackVersion>(), 8);
        assert_eq!(K_DL_CPU, 1);
        assert_eq!(K_DL_FLOAT, 2);
    }
}
```
(If a size assertion fails on the dev arch, FIX the assertion to the real value AND record why in the commit — the goal is to LOCK whatever the correct 64-bit layout is, not to force a guessed number. On all LP64 targets these values hold.)

- [ ] **Step 5: rewrite `src/lib.rs`**:
```rust
//! Portable DLPack bindings: raw FFI ([`ffi`]), safe builders ([`safe`]),
//! and optional pyo3 capsule glue (feature `pyo3`).
pub mod ffi;
pub mod safe;
#[cfg(feature = "pyo3")]
pub mod pyo3_glue;
```
(Create a minimal `src/safe.rs` stub — `//! placeholder` — so it compiles; filled in Task 2.)

- [ ] **Step 6: build + test (default features only)**
Run: `cargo build && cargo test --no-default-features`
Expected: PASS (ffi abi_layout test). Also `cargo clippy --no-default-features -- -D warnings`.

- [ ] **Step 7: commit**
```
git add -A
git commit -m "refactor!: portable hand-written FFI; drop bindgen/submodule/x86 dump"
```

---

### Task 2: Safe wrappers + ownership/deleter

**Files:** rewrite `src/safe.rs`. **Consumes:** `ffi` (Task 1). **Produces:** `TensorInfo`, dtype/device ctors, `pack`/`pack_versioned`.

- [ ] **Step 1: write `src/safe.rs`** — `TensorInfo` + ctors + the `ManagedContext`/deleter (the keep-alive crux):
```rust
//! Safe builders over the DLPack ABI, with ownership-correct deleters.
use crate::ffi::*;
use core::ffi::c_void;

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub dtype: DLDataType,
    pub shape: Vec<i64>,
    pub strides: Option<Vec<i64>>,
    pub byte_offset: u64,
}
impl TensorInfo {
    pub fn contiguous(data: *mut c_void, device: DLDevice, dtype: DLDataType, shape: Vec<i64>) -> Self {
        Self { data, device, dtype, shape, strides: None, byte_offset: 0 }
    }
    pub fn strided(data: *mut c_void, device: DLDevice, dtype: DLDataType, shape: Vec<i64>, strides: Vec<i64>) -> Self {
        Self { data, device, dtype, shape, strides: Some(strides), byte_offset: 0 }
    }
    pub fn with_byte_offset(mut self, off: u64) -> Self { self.byte_offset = off; self }
}

pub fn cpu_device() -> DLDevice { DLDevice { device_type: K_DL_CPU, device_id: 0 } }
pub fn cuda_device(id: i32) -> DLDevice { DLDevice { device_type: K_DL_CUDA, device_id: id } }
pub fn dtype_u8() -> DLDataType { DLDataType { code: K_DL_UINT, bits: 8, lanes: 1 } }
pub fn dtype_u16() -> DLDataType { DLDataType { code: K_DL_UINT, bits: 16, lanes: 1 } }
pub fn dtype_i32() -> DLDataType { DLDataType { code: K_DL_INT, bits: 32, lanes: 1 } }
pub fn dtype_i64() -> DLDataType { DLDataType { code: K_DL_INT, bits: 64, lanes: 1 } }
pub fn dtype_f32() -> DLDataType { DLDataType { code: K_DL_FLOAT, bits: 32, lanes: 1 } }
pub fn dtype_f64() -> DLDataType { DLDataType { code: K_DL_FLOAT, bits: 64, lanes: 1 } }
pub fn dtype_bool() -> DLDataType { DLDataType { code: K_DL_BOOL, bits: 8, lanes: 1 } }

// Owns the keep-alive value + the shape/strides backing the tensor's pointers.
struct ManagedContext<T> {
    _keepalive: T,
    shape: Vec<i64>,
    strides: Option<Vec<i64>>,
}

/// Pack `keepalive` + `info` into a heap `DLManagedTensor`. The returned pointer
/// is owned by the caller (or the consumer that takes the capsule); it is freed
/// only by invoking its `deleter`.
pub fn pack<T: 'static>(keepalive: T, info: TensorInfo) -> *mut DLManagedTensor {
    // Box the context so shape/strides have a stable address for the tensor's pointers.
    let mut ctx = Box::new(ManagedContext { _keepalive: keepalive, shape: info.shape, strides: info.strides });
    let shape_ptr = ctx.shape.as_mut_ptr();
    let ndim = ctx.shape.len() as i32;
    let strides_ptr = match ctx.strides.as_mut() { Some(s) => s.as_mut_ptr(), None => core::ptr::null_mut() };
    let dl_tensor = DLTensor {
        data: info.data, device: info.device, ndim, dtype: info.dtype,
        shape: shape_ptr, strides: strides_ptr, byte_offset: info.byte_offset,
    };
    let ctx_ptr = Box::into_raw(ctx) as *mut c_void;
    let mt = Box::new(DLManagedTensor { dl_tensor, manager_ctx: ctx_ptr, deleter: Some(deleter::<T>) });
    Box::into_raw(mt)
}

unsafe extern "C" fn deleter<T: 'static>(mt: *mut DLManagedTensor) {
    if mt.is_null() { return; }
    // SAFETY: mt was produced by `pack::<T>` (Box::into_raw); reclaim both boxes.
    let mt = Box::from_raw(mt);
    if !mt.manager_ctx.is_null() {
        drop(Box::from_raw(mt.manager_ctx as *mut ManagedContext<T>));
    }
    // mt (DLManagedTensor box) dropped here.
}

/// Versioned (DLPack 1.0) variant.
pub fn pack_versioned<T: 'static>(keepalive: T, info: TensorInfo, flags: u64) -> *mut DLManagedTensorVersioned {
    let mut ctx = Box::new(ManagedContext { _keepalive: keepalive, shape: info.shape, strides: info.strides });
    let shape_ptr = ctx.shape.as_mut_ptr();
    let ndim = ctx.shape.len() as i32;
    let strides_ptr = match ctx.strides.as_mut() { Some(s) => s.as_mut_ptr(), None => core::ptr::null_mut() };
    let dl_tensor = DLTensor { data: info.data, device: info.device, ndim, dtype: info.dtype, shape: shape_ptr, strides: strides_ptr, byte_offset: info.byte_offset };
    let ctx_ptr = Box::into_raw(ctx) as *mut c_void;
    let mt = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: ctx_ptr, deleter: Some(deleter_versioned::<T>), flags, dl_tensor,
    });
    Box::into_raw(mt)
}

unsafe extern "C" fn deleter_versioned<T: 'static>(mt: *mut DLManagedTensorVersioned) {
    if mt.is_null() { return; }
    let mt = Box::from_raw(mt);
    if !mt.manager_ctx.is_null() {
        drop(Box::from_raw(mt.manager_ctx as *mut ManagedContext<T>));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use std::cell::Cell;
    #[test]
    fn pack_then_delete_drops_keepalive_once() {
        // keep-alive whose Drop flips a flag — proves no leak / no double free.
        struct Guard(Rc<Cell<u32>>);
        impl Drop for Guard { fn drop(&mut self) { self.0.set(self.0.get() + 1); } }
        let flag = Rc::new(Cell::new(0u32));
        let mut buf = vec![1.0f32, 2.0, 3.0, 4.0];
        let info = TensorInfo::contiguous(buf.as_mut_ptr() as *mut _, cpu_device(), dtype_f32(), vec![2, 2]);
        let mt = pack(Guard(flag.clone()), info);
        unsafe {
            // shape pointer is valid and matches
            assert_eq!((*mt).dl_tensor.ndim, 2);
            assert_eq!(*(*mt).dl_tensor.shape.add(0), 2);
            let del = (*mt).deleter.unwrap();
            del(mt);
        }
        assert_eq!(flag.get(), 1, "keepalive dropped exactly once");
        drop(buf);
    }
}
```

- [ ] **Step 2: build + test**
Run: `cargo test --no-default-features` and `cargo clippy --no-default-features -- -D warnings`
Expected: PASS (abi_layout + pack_then_delete_drops_keepalive_once).

- [ ] **Step 3: commit**
```
git add -A && git commit -m "feat(safe): TensorInfo + ownership-correct pack/deleter (legacy + versioned)"
```

---

### Task 3: pyo3 glue (feature) + CI + README + tag

**Files:** create `src/pyo3_glue.rs`; rewrite `.github/workflows/ci.yml`, `README.md`. **Consumes:** `safe` (Task 2).

- [ ] **Step 1: write `src/pyo3_glue.rs`** (`#![cfg(feature = "pyo3")]` via the `lib.rs` gate) — `IntoDLPack` export + `PyTensor` import. Model the capsule lifecycle on pyo3-dlpack:
  - `into_capsule`: `safe::pack(keepalive, info)` → `PyCapsule::new_with_destructor(py, mt_ptr, name="dltensor", destructor)`. Destructor: if capsule name is still `"dltensor"` (consumer never consumed), call the tensor's `deleter`; if `"used_dltensor"`, do nothing.
  - `into_capsule_versioned`: name `"dltensor_versioned"`, `pack_versioned`, READ_ONLY flag optional.
  - `PyTensor::from_pyany(py, obj)`: `obj.call_method0("__dlpack__")` → downcast `PyCapsule` → validate name → read the `*mut DLManagedTensor[Versioned]` → **rename capsule to `"used_dltensor"`** → store `Py<PyCapsule>` + ptr. Accessors: `shape/strides/dtype/device/data_ptr/byte_offset/is_contiguous/numel/nbytes`. `Drop`: call the producer's `deleter`.
  - Provide a concrete `from_pyany` + accessors; exact pyo3 0.28 capsule API: use `pyo3::types::PyCapsule`, `PyCapsule::new_with_destructor`, `capsule.name()`, `capsule.set_name(...)`, `capsule.pointer()`.
  If a precise pyo3 0.28 capsule call differs, adapt to what compiles (this is the one place to consult pyo3 docs) — keep the rename-on-consume + deleter-on-drop semantics intact.

- [ ] **Step 2: build both feature sets + clippy**
Run: `cargo build --features pyo3 && cargo clippy --features pyo3 -- -D warnings && cargo test --no-default-features`
Expected: PASS. (A full Python round-trip is deferred to kornia-py task #24; here just compile + the non-Python tests.)

- [ ] **Step 3: rewrite `.github/workflows/ci.yml`** — trigger on push/PR; matrix over `{features: [--no-default-features, --features pyo3]}` and `{os: ubuntu-latest, aarch64}` (use `runs-on: ubuntu-24.04-arm` or QEMU); steps: checkout (NO submodule), `cargo fmt --check`, `cargo clippy ... -- -D warnings`, `cargo test`. Remove the auto-commit-bindgen and auto-publish steps.

- [ ] **Step 4: rewrite `README.md`** — what it is; install (`git`/tag dep + crates.io later); three usage snippets (raw `ffi`, `safe::pack`, pyo3 `__dlpack__`); feature flags; license.

- [ ] **Step 5: commit + tag**
```
git add -A && git commit -m "feat(pyo3): __dlpack__ capsule glue (feature) + aarch64 CI + README"
git tag v0.2.0
```
(Push + tag push happens after review, on user confirmation — outward-facing.)

---

## Self-Review checklist
- No `bindgen`/`build.rs`/submodule/`dlpack.rs` remain (`git ls-files | grep -E 'bindgen|build.rs|dlpack.rs|.gitmodules'` empty).
- `cargo test --no-default-features` and `cargo build --features pyo3` both pass; clippy `-D warnings` clean for both.
- Deleter test proves keep-alive dropped exactly once.
- FFI struct names/fields/order match dmlc/dlpack; ABI size asserts pass.
- Consumer (kornia-py #24) can depend via `{ git, tag = "v0.2.0" }`.
