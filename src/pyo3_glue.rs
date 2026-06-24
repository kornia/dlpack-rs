//! pyo3 capsule glue: `IntoDLPack` (export) and `PyTensor` (import).
//!
//! Feature-gated: only compiled when `--features pyo3` is active.
//!
//! # Export — `into_capsule` / `into_capsule_versioned`
//!
//! Wraps a `*mut DLManagedTensor[Versioned]` (produced by [`crate::safe::pack`]) inside a
//! Python `PyCapsule` named `"dltensor"` / `"dltensor_versioned"`.  The capsule's C destructor
//! inspects the current capsule name:
//! - still `"dltensor"` → producer was never consumed → call the DLPack `deleter`.
//! - `"used_dltensor"` → consumer already renamed it → do nothing (consumer owns the pointer).
//!
//! # Import — `PyTensor::from_pyany`
//!
//! Calls `obj.__dlpack__()`, validates the capsule name, reads the raw pointer,
//! **renames the capsule to `"used_dltensor"`** (preventing the capsule destructor from
//! double-freeing), and stores both the `Py<PyCapsule>` (keep-alive) and the pointer.
//! `Drop` calls the producer's own `deleter`.

use crate::ffi::{DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned};
use crate::safe::{self, TensorInfo};
use pyo3::ffi as pyffi;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};
use std::ffi::{c_void, CStr};
use std::ptr::NonNull;

// Static C-string literals used as capsule names.
// Their addresses are stable for the entire program lifetime, which is required
// by PyCapsule_SetName (the name pointer must remain valid while the capsule lives).
const NAME_DLTENSOR: &CStr = c"dltensor";
const NAME_DLTENSOR_VERSIONED: &CStr = c"dltensor_versioned";
const NAME_USED_DLTENSOR: &CStr = c"used_dltensor";
const NAME_USED_DLTENSOR_VERSIONED: &CStr = c"used_dltensor_versioned";

// ──────────────────────────────────────────────────────────────────────────────
// Capsule destructors (extern "C", called by Python GC)
// ──────────────────────────────────────────────────────────────────────────────

/// Capsule destructor for the legacy `"dltensor"` capsule.
///
/// Only fires the DLPack deleter if the capsule was never consumed (name still `"dltensor"`).
/// If a consumer renamed it to `"used_dltensor"` via `PyTensor::from_pyany`, we do nothing
/// — the consumer's `Drop` will call the deleter instead.
unsafe extern "C" fn capsule_destructor_legacy(capsule: *mut pyffi::PyObject) {
    // SAFETY: Python calls capsule destructors with the GIL already held; do NOT call
    // Python::attach or attempt to re-acquire the GIL here (would deadlock).
    // `capsule` is a valid PyCapsule* being finalized.
    let name_ptr = unsafe { pyffi::PyCapsule_GetName(capsule) };

    // Check if the name is still "dltensor" (i.e. not renamed to "used_dltensor").
    // A null name_ptr means unnamed; treat as consumed (do nothing).
    if name_ptr.is_null() {
        return;
    }
    // SAFETY: name_ptr came from PyCapsule_GetName and is a valid C string while the capsule lives.
    let name = unsafe { CStr::from_ptr(name_ptr) };
    if name != NAME_DLTENSOR {
        // Already renamed to "used_dltensor" — consumer owns the pointer; do nothing.
        return;
    }

    // SAFETY: capsule was constructed in `into_capsule` with a *mut DLManagedTensor payload.
    let ptr = unsafe { pyffi::PyCapsule_GetPointer(capsule, name_ptr) };
    if ptr.is_null() {
        return;
    }
    let mt = ptr as *mut DLManagedTensor;
    // SAFETY: mt is a valid *mut DLManagedTensor allocated by safe::pack; calling its deleter
    // is exactly the contract we signed — it reclaims the two Boxes (DLManagedTensor +
    // ManagedContext<T>). We only reach here if it was never consumed.
    if let Some(del) = unsafe { (*mt).deleter } {
        unsafe { del(mt) };
    }
}

/// Capsule destructor for the versioned `"dltensor_versioned"` capsule.
unsafe extern "C" fn capsule_destructor_versioned(capsule: *mut pyffi::PyObject) {
    // SAFETY: Python calls capsule destructors with the GIL already held; do NOT call
    // Python::attach or attempt to re-acquire the GIL here (would deadlock).
    // `capsule` is a valid PyCapsule* being finalized.
    let name_ptr = unsafe { pyffi::PyCapsule_GetName(capsule) };
    if name_ptr.is_null() {
        return;
    }
    // SAFETY: name_ptr is valid while the capsule lives.
    let name = unsafe { CStr::from_ptr(name_ptr) };
    if name != NAME_DLTENSOR_VERSIONED {
        return;
    }

    let ptr = unsafe { pyffi::PyCapsule_GetPointer(capsule, name_ptr) };
    if ptr.is_null() {
        return;
    }
    let mt = ptr as *mut DLManagedTensorVersioned;
    // SAFETY: mt is a valid *mut DLManagedTensorVersioned from safe::pack_versioned.
    if let Some(del) = unsafe { (*mt).deleter } {
        unsafe { del(mt) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Export: IntoDLPack trait
// ──────────────────────────────────────────────────────────────────────────────

/// Trait for types that can export themselves as a DLPack capsule.
///
/// Implement this on your buffer wrapper, then call `into_capsule` or
/// `into_capsule_versioned` to hand a zero-copy tensor to Python.
pub trait IntoDLPack: Sized + Send + 'static {
    /// Return the [`TensorInfo`] describing this tensor's layout.
    fn tensor_info(&self) -> TensorInfo;

    /// Pack `self` into a legacy `"dltensor"` PyCapsule.
    ///
    /// Ownership transfers to Python: when Python GC collects the capsule
    /// (and no consumer renamed it to `"used_dltensor"`), the capsule destructor
    /// calls `DLManagedTensor.deleter`, which drops `self`.
    fn into_capsule(self, py: Python<'_>) -> PyResult<Bound<'_, PyCapsule>> {
        let info = self.tensor_info();
        let mt = safe::pack(self, info);

        // SAFETY: mt is a freshly heap-allocated *mut DLManagedTensor (non-null).
        // We convert it to NonNull<c_void> and hand it to the capsule. Ownership
        // transfers: if the capsule is GC'd without being consumed, the C destructor
        // will call mt's own deleter. If consumed (renamed), Drop of PyTensor calls it.
        let ptr = NonNull::new(mt as *mut c_void).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("pack returned null")
        })?;

        unsafe {
            PyCapsule::new_with_pointer_and_destructor(
                py,
                ptr,
                NAME_DLTENSOR,
                Some(capsule_destructor_legacy),
            )
        }
    }

    /// Pack `self` into a versioned `"dltensor_versioned"` PyCapsule (DLPack 1.0).
    fn into_capsule_versioned(self, py: Python<'_>, flags: u64) -> PyResult<Bound<'_, PyCapsule>> {
        let info = self.tensor_info();
        let mt = safe::pack_versioned(self, info, flags);

        // SAFETY: same as into_capsule; mt is a valid *mut DLManagedTensorVersioned.
        let ptr = NonNull::new(mt as *mut c_void).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("pack_versioned returned null")
        })?;

        unsafe {
            PyCapsule::new_with_pointer_and_destructor(
                py,
                ptr,
                NAME_DLTENSOR_VERSIONED,
                Some(capsule_destructor_versioned),
            )
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Import: PyTensor
// ──────────────────────────────────────────────────────────────────────────────

/// Describes which DLPack protocol variant was used.
///
/// Named `DLPackVariant` (not `DLPackVersion`) to avoid shadowing the C-ABI struct
/// `ffi::DLPackVersion { major, minor }` whose name is mandated by the DLPack spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLPackVariant {
    /// Legacy `DLManagedTensor` capsule named `"dltensor"`.
    Legacy,
    /// Versioned `DLManagedTensorVersioned` capsule named `"dltensor_versioned"`.
    Versioned,
}

/// An imported DLPack tensor, borrowed from a Python object via `__dlpack__()`.
///
/// - Holds a `Py<PyCapsule>` (keep-alive so GC does not collect the capsule while we use the
///   pointer).
/// - Calls the producer's `deleter` on `Drop`.
///
/// # Lifecycle
/// 1. `from_pyany` calls `obj.__dlpack__()` → Python capsule.
/// 2. Validates the capsule name (`"dltensor"` or `"dltensor_versioned"`).
/// 3. Reads the raw pointer via `PyCapsule_GetPointer`.
/// 4. **Renames** the capsule to `"used_dltensor"` / `"used_dltensor_versioned"` so the
///    capsule's own C destructor will NOT call the deleter again.
/// 5. On `Drop`, calls the producer's `deleter` exactly once.
pub struct PyTensor {
    /// Keep the capsule alive so Python does not GC it while we hold `ptr`.
    _capsule: Py<PyCapsule>,
    /// Raw pointer to the DLManagedTensor (legacy) or DLManagedTensorVersioned.
    ptr: *mut c_void,
    /// Which variant we consumed.
    version: DLPackVariant,
}

// SAFETY: PyTensor owns the pointed-to data exclusively; raw pointer is not Clone/Copy.
// The pointer came from safe::pack[_versioned] which requires T: Send + 'static.
unsafe impl Send for PyTensor {}

impl PyTensor {
    /// Import a DLPack tensor from any Python object that implements `__dlpack__()`.
    ///
    /// Returns an error if `__dlpack__()` is not present, returns a non-capsule object,
    /// or the capsule name is not one of the expected DLPack names.
    pub fn from_pyany(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // 1. Call __dlpack__() — producer fills and returns the capsule.
        let capsule_obj = obj.call_method0("__dlpack__")?;

        // 2. Downcast to PyCapsule.
        let capsule: Bound<'_, PyCapsule> = capsule_obj.cast_into()?;

        // 3. Determine the DLPack variant from the capsule name.
        let cap_name = capsule.name()?;
        let name_cstr: &CStr = match &cap_name {
            Some(n) => {
                // SAFETY: n is the capsule name; it is valid as long as `capsule` is alive
                // and we do not mutate the name before reading it here.
                unsafe { n.as_cstr() }
            }
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "DLPack capsule has no name",
                ));
            }
        };

        let version = if name_cstr == NAME_DLTENSOR {
            DLPackVariant::Legacy
        } else if name_cstr == NAME_DLTENSOR_VERSIONED {
            DLPackVariant::Versioned
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unexpected DLPack capsule name: expected 'dltensor' or 'dltensor_versioned', got {:?}",
                name_cstr
            )));
        };

        // 4. Read the raw pointer (name-checked).
        let nn_ptr: NonNull<c_void> = capsule.pointer_checked(Some(name_cstr))?;

        // 5. Rename the capsule → prevent the C destructor from calling deleter on GC.
        let new_name = match version {
            DLPackVariant::Legacy => NAME_USED_DLTENSOR.as_ptr(),
            DLPackVariant::Versioned => NAME_USED_DLTENSOR_VERSIONED.as_ptr(),
        };
        // SAFETY: capsule.as_ptr() is a valid PyCapsule* with thread attached.
        // new_name is a &'static CStr pointer so it outlives the capsule.
        let rc = unsafe { pyffi::PyCapsule_SetName(capsule.as_ptr(), new_name) };
        if rc != 0 {
            return Err(PyErr::fetch(py));
        }

        // 6. Keep the capsule alive via Py<PyCapsule>.
        let owned_cap: Py<PyCapsule> = capsule.into();

        Ok(PyTensor {
            _capsule: owned_cap,
            ptr: nn_ptr.as_ptr(),
            version,
        })
    }

    /// Which DLPack variant was imported.
    pub fn dlpack_version(&self) -> DLPackVariant {
        self.version
    }

    // ── Internal helpers to borrow the DLTensor ────────────────────────────

    /// Reference to the inner `DLTensor` (legacy variant).
    ///
    /// # Safety
    /// `ptr` is valid for as long as `self` is alive; caller must not call the deleter.
    fn dl_tensor(&self) -> &crate::ffi::DLTensor {
        match self.version {
            DLPackVariant::Legacy => {
                // SAFETY: ptr is a valid *mut DLManagedTensor produced by safe::pack.
                unsafe { &(*(self.ptr as *const DLManagedTensor)).dl_tensor }
            }
            DLPackVariant::Versioned => {
                // SAFETY: ptr is a valid *mut DLManagedTensorVersioned produced by safe::pack_versioned.
                unsafe { &(*(self.ptr as *const DLManagedTensorVersioned)).dl_tensor }
            }
        }
    }

    // ── Public accessors ────────────────────────────────────────────────────

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dl_tensor().ndim as usize
    }

    /// Shape as a slice. Valid for the lifetime of `self`.
    pub fn shape(&self) -> &[i64] {
        let t = self.dl_tensor();
        // SAFETY: shape is a valid pointer to ndim i64 values, stable until deleter is called.
        unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) }
    }

    /// Strides as a slice, or `None` for compact row-major layout.
    pub fn strides(&self) -> Option<&[i64]> {
        let t = self.dl_tensor();
        if t.strides.is_null() {
            None
        } else {
            // SAFETY: strides is valid (non-null) pointer to ndim i64 values.
            Some(unsafe { std::slice::from_raw_parts(t.strides, t.ndim as usize) })
        }
    }

    /// Data type descriptor.
    pub fn dtype(&self) -> DLDataType {
        self.dl_tensor().dtype
    }

    /// Device descriptor.
    pub fn device(&self) -> DLDevice {
        self.dl_tensor().device
    }

    /// Raw data pointer (untyped).
    ///
    /// Note: the first element is at `data_ptr()` advanced by `byte_offset()` bytes.
    pub fn data_ptr(&self) -> *mut c_void {
        self.dl_tensor().data
    }

    /// Byte offset from `data_ptr` to the first valid element.
    pub fn byte_offset(&self) -> u64 {
        self.dl_tensor().byte_offset
    }

    /// Total number of elements (product of shape dims).
    pub fn numel(&self) -> usize {
        self.shape().iter().map(|&d| d as usize).product()
    }

    /// Size of one element in bytes (`bits / 8 * lanes`).
    pub fn element_size(&self) -> usize {
        let dt = self.dtype();
        (dt.bits as usize).div_ceil(8) * dt.lanes as usize
    }

    /// Total byte size of the tensor data.
    pub fn nbytes(&self) -> usize {
        self.numel() * self.element_size()
    }

    /// `true` if the tensor is stored contiguously in row-major order.
    pub fn is_contiguous(&self) -> bool {
        match self.strides() {
            None => true,
            Some(strides) => {
                let shape = self.shape();
                let ndim = shape.len();
                if ndim == 0 {
                    return true;
                }
                let mut expected = 1i64;
                for i in (0..ndim).rev() {
                    if strides[i] != expected {
                        return false;
                    }
                    expected *= shape[i];
                }
                true
            }
        }
    }
}

impl Drop for PyTensor {
    fn drop(&mut self) {
        // SAFETY/GIL: the producer's deleter may re-enter Python (e.g. torch frees a
        // TensorImpl), so it MUST run under the GIL. Drop can fire off-GIL (Rust-side),
        // so we acquire it here. Python::attach is reentrant-safe when already held,
        // and a no-op cost-wise when the GIL is already owned by this thread.
        //
        // The capsule destructor has been neutralised (name renamed to "used_*"),
        // so this is the sole deallocation path — exactly once.
        let ptr = self.ptr;
        let version = self.version;
        Python::attach(|_py| {
            match version {
                DLPackVariant::Legacy => {
                    let mt = ptr as *mut DLManagedTensor;
                    // SAFETY: mt was produced by safe::pack and renamed in from_pyany to
                    // prevent double-free. This Drop is the one and only call to the deleter.
                    if let Some(del) = unsafe { (*mt).deleter } {
                        unsafe { del(mt) };
                    }
                }
                DLPackVariant::Versioned => {
                    let mt = ptr as *mut DLManagedTensorVersioned;
                    // SAFETY: as above, for the versioned variant.
                    if let Some(del) = unsafe { (*mt).deleter } {
                        unsafe { del(mt) };
                    }
                }
            }
        });
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests (compile-time only; full round-trip needs a Python interpreter)
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safe::{cpu_device, dtype_f32};

    /// Smoke-test that `into_capsule` compiles and the IntoDLPack trait is object-safe enough
    /// for our purposes. No Python interpreter needed for this compile check.
    struct DummyBuffer {
        data: Vec<f32>,
    }

    impl IntoDLPack for DummyBuffer {
        fn tensor_info(&self) -> TensorInfo {
            TensorInfo::contiguous(
                self.data.as_ptr() as *mut c_void,
                cpu_device(),
                dtype_f32(),
                vec![self.data.len() as i64],
            )
        }
    }

    // This test only verifies that the types and trait impl compile.
    // A live capsule round-trip would require pyo3::prepare_freethreaded_python().
    #[test]
    fn trait_impl_compiles() {
        let _buf = DummyBuffer {
            data: vec![1.0f32, 2.0, 3.0],
        };
        // Just check that tensor_info() works without panicking.
        let info = _buf.tensor_info();
        assert_eq!(info.shape, vec![3i64]);
    }
}
