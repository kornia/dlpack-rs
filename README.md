# dlpack-rs

Portable Rust bindings for the [DLPack](https://dmlc.github.io/dlpack/latest/) tensor protocol.

- **`dlpack_rs::ffi`** — hand-written `#[repr(C)]` ABI types (`DLTensor`, `DLManagedTensor`,
  `DLManagedTensorVersioned`, `DLDevice`, `DLDataType`, `DLPackVersion`).
  No `bindgen`, no C submodule, no build script.
- **`dlpack_rs::safe`** — ownership-correct `pack` / `pack_versioned` builders that wrap a
  Rust keepalive + shape/strides Vecs in a heap-allocated `DLManagedTensor[Versioned]` with a
  correct `deleter` callback.
- **`dlpack_rs::pyo3_glue`** (feature `pyo3`) — `IntoDLPack` export trait + `PyTensor` import
  struct; implements the `__dlpack__` capsule lifecycle (rename-on-consume + deleter-on-drop).

## Installation

Add to `Cargo.toml`:

```toml
# no pyo3 glue (pure Rust, no Python dependency):
dlpack-rs = { git = "https://github.com/kornia/dlpack-rs", tag = "v0.2.0" }

# with pyo3 glue:
dlpack-rs = { git = "https://github.com/kornia/dlpack-rs", tag = "v0.2.0", features = ["pyo3"] }
```

## Usage

### Raw FFI

```rust
use dlpack_rs::ffi::{DLDataType, DLDevice, DLTensor, K_DL_CPU, K_DL_FLOAT};

let device = DLDevice { device_type: K_DL_CPU, device_id: 0 };
let dtype  = DLDataType { code: K_DL_FLOAT, bits: 32, lanes: 1 };
```

### Safe builder (`safe::pack`)

```rust
use dlpack_rs::safe::{pack, cpu_device, dtype_f32, TensorInfo};

let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
let info = TensorInfo::contiguous(
    data.as_mut_ptr() as *mut _,
    cpu_device(),
    dtype_f32(),
    vec![2, 2],
);

// `data` is the keepalive: it will be dropped when the DLPack deleter fires.
let mt: *mut dlpack_rs::ffi::DLManagedTensor = pack(data, info);

// Hand `mt` to a consumer (e.g. Python via a capsule).
// The consumer calls mt.deleter(mt) when done; that drops the keepalive.
```

### pyo3 `__dlpack__` (feature `pyo3`)

**Export** — give a Python caller a zero-copy view of Rust data:

```rust
use dlpack_rs::pyo3_glue::IntoDLPack;
use dlpack_rs::safe::{cpu_device, dtype_f32, TensorInfo};
use pyo3::prelude::*;
use std::ffi::c_void;

struct MyTensor(Vec<f32>);

impl IntoDLPack for MyTensor {
    fn tensor_info(&self) -> TensorInfo {
        TensorInfo::contiguous(
            self.0.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            vec![self.0.len() as i64],
        )
    }
}

Python::attach(|py| {
    let t = MyTensor(vec![1.0, 2.0, 3.0]);
    let capsule = t.into_capsule(py).unwrap();
    // pass `capsule` to Python as the return value of `__dlpack__()`
});
```

**Import** — consume a capsule from any Python object that implements `__dlpack__()`:

```rust
use dlpack_rs::pyo3_glue::PyTensor;
use pyo3::prelude::*;

Python::attach(|py| {
    let obj: Bound<'_, PyAny> = /* ... numpy array, torch tensor, etc. */ todo!();
    let tensor = PyTensor::from_pyany(py, &obj).unwrap();

    println!("shape  = {:?}", tensor.shape());
    println!("dtype  = {:?}", tensor.dtype());
    println!("device = {:?}", tensor.device());
    println!("nbytes = {}", tensor.nbytes());

    // When `tensor` is dropped, the producer's DLPack deleter is called exactly once.
});
```

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `pyo3`  | no      | Enables `pyo3_glue` (requires Python headers at build time) |

Without any features (`--no-default-features`), the crate is a pure `no_std`-compatible FFI +
safe-builder layer with zero non-`std` dependencies.

## License

Apache-2.0 — see [LICENSE](LICENSE).
