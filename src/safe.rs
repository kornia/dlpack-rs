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
    pub fn contiguous(
        data: *mut c_void,
        device: DLDevice,
        dtype: DLDataType,
        shape: Vec<i64>,
    ) -> Self {
        Self {
            data,
            device,
            dtype,
            shape,
            strides: None,
            byte_offset: 0,
        }
    }
    pub fn strided(
        data: *mut c_void,
        device: DLDevice,
        dtype: DLDataType,
        shape: Vec<i64>,
        strides: Vec<i64>,
    ) -> Self {
        Self {
            data,
            device,
            dtype,
            shape,
            strides: Some(strides),
            byte_offset: 0,
        }
    }
    pub fn with_byte_offset(mut self, off: u64) -> Self {
        self.byte_offset = off;
        self
    }
}

pub fn cpu_device() -> DLDevice {
    DLDevice {
        device_type: K_DL_CPU,
        device_id: 0,
    }
}
pub fn cuda_device(id: i32) -> DLDevice {
    DLDevice {
        device_type: K_DL_CUDA,
        device_id: id,
    }
}
pub fn dtype_u8() -> DLDataType {
    DLDataType {
        code: K_DL_UINT,
        bits: 8,
        lanes: 1,
    }
}
pub fn dtype_u16() -> DLDataType {
    DLDataType {
        code: K_DL_UINT,
        bits: 16,
        lanes: 1,
    }
}
pub fn dtype_i32() -> DLDataType {
    DLDataType {
        code: K_DL_INT,
        bits: 32,
        lanes: 1,
    }
}
pub fn dtype_i64() -> DLDataType {
    DLDataType {
        code: K_DL_INT,
        bits: 64,
        lanes: 1,
    }
}
pub fn dtype_f32() -> DLDataType {
    DLDataType {
        code: K_DL_FLOAT,
        bits: 32,
        lanes: 1,
    }
}
pub fn dtype_f64() -> DLDataType {
    DLDataType {
        code: K_DL_FLOAT,
        bits: 64,
        lanes: 1,
    }
}
pub fn dtype_bool() -> DLDataType {
    DLDataType {
        code: K_DL_BOOL,
        bits: 8,
        lanes: 1,
    }
}

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
    let mut ctx = Box::new(ManagedContext {
        _keepalive: keepalive,
        shape: info.shape,
        strides: info.strides,
    });
    let shape_ptr = ctx.shape.as_mut_ptr();
    let ndim = ctx.shape.len() as i32;
    let strides_ptr = match ctx.strides.as_mut() {
        Some(s) => s.as_mut_ptr(),
        None => core::ptr::null_mut(),
    };
    let dl_tensor = DLTensor {
        data: info.data,
        device: info.device,
        ndim,
        dtype: info.dtype,
        shape: shape_ptr,
        strides: strides_ptr,
        byte_offset: info.byte_offset,
    };
    let ctx_ptr = Box::into_raw(ctx) as *mut c_void;
    // SAFETY: ctx_ptr is a valid Box<ManagedContext<T>> cast to *mut c_void;
    // shape_ptr/strides_ptr point into the heap-allocated Vecs inside that box,
    // which remain valid until the deleter drops the box.
    let mt = Box::new(DLManagedTensor {
        dl_tensor,
        manager_ctx: ctx_ptr,
        deleter: Some(deleter::<T>),
    });
    Box::into_raw(mt)
}

unsafe extern "C" fn deleter<T: 'static>(mt: *mut DLManagedTensor) {
    if mt.is_null() {
        return;
    }
    // SAFETY: mt was produced by `pack::<T>` (Box::into_raw); reclaim both boxes.
    // The manager_ctx pointer holds a Box<ManagedContext<T>> cast to *mut c_void.
    // We reconstruct and drop them here — this is the only call site (called exactly once
    // by the DLPack consumer), so there is no double-free.
    let mt = Box::from_raw(mt);
    if !mt.manager_ctx.is_null() {
        drop(Box::from_raw(mt.manager_ctx as *mut ManagedContext<T>));
    }
    // mt (DLManagedTensor box) dropped here.
}

/// Versioned (DLPack 1.0) variant.
pub fn pack_versioned<T: 'static>(
    keepalive: T,
    info: TensorInfo,
    flags: u64,
) -> *mut DLManagedTensorVersioned {
    let mut ctx = Box::new(ManagedContext {
        _keepalive: keepalive,
        shape: info.shape,
        strides: info.strides,
    });
    let shape_ptr = ctx.shape.as_mut_ptr();
    let ndim = ctx.shape.len() as i32;
    let strides_ptr = match ctx.strides.as_mut() {
        Some(s) => s.as_mut_ptr(),
        None => core::ptr::null_mut(),
    };
    let dl_tensor = DLTensor {
        data: info.data,
        device: info.device,
        ndim,
        dtype: info.dtype,
        shape: shape_ptr,
        strides: strides_ptr,
        byte_offset: info.byte_offset,
    };
    let ctx_ptr = Box::into_raw(ctx) as *mut c_void;
    // SAFETY: Same as pack — ctx_ptr owns the shape/strides Vecs whose pointers are in dl_tensor.
    // The deleter_versioned fn is the sole owner-reclaimer.
    let mt = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: ctx_ptr,
        deleter: Some(deleter_versioned::<T>),
        flags,
        dl_tensor,
    });
    Box::into_raw(mt)
}

unsafe extern "C" fn deleter_versioned<T: 'static>(mt: *mut DLManagedTensorVersioned) {
    if mt.is_null() {
        return;
    }
    // SAFETY: mt was produced by `pack_versioned::<T>` (Box::into_raw); reclaim both boxes.
    let mt = Box::from_raw(mt);
    if !mt.manager_ctx.is_null() {
        drop(Box::from_raw(mt.manager_ctx as *mut ManagedContext<T>));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;
    #[test]
    fn pack_then_delete_drops_keepalive_once() {
        // keep-alive whose Drop flips a flag — proves no leak / no double free.
        struct Guard(Rc<Cell<u32>>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let flag = Rc::new(Cell::new(0u32));
        let mut buf = vec![1.0f32, 2.0, 3.0, 4.0];
        let info = TensorInfo::contiguous(
            buf.as_mut_ptr() as *mut _,
            cpu_device(),
            dtype_f32(),
            vec![2, 2],
        );
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
