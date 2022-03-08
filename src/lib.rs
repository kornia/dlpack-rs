pub mod dlpack;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::ffi::{CStr, CString};
use std::os::raw::{c_int, c_void};

type NDArray = ndarray::ArrayD<f32>;

fn array_to_dl_tensor(arr: &Box<NDArray>) -> dlpack::DLTensor {
    let device = dlpack::DLDevice {
        device_type: dlpack::DLDeviceType_kDLCPU,
        device_id: 0,
    };

    dlpack::DLTensor {
        data: arr.as_ptr() as *const c_void as *mut c_void,
        device: device,
        ndim: arr.ndim() as c_int,
        dtype: dlpack::DLDataType {
            code: dlpack::DLDataTypeCode_kDLFloat as u8,
            bits: 32,
            lanes: 1,
        },
        shape: arr.shape().as_ptr() as *const i64 as *mut i64,
        strides: arr.strides().as_ptr() as *const i64 as *mut i64,
        byte_offset: 0,
    }
}

unsafe extern "C" fn deleter(x: *mut dlpack::DLManagedTensor) {
    println!("DLManagedTensor deleter");

    // let ctx: *mut ArrayDLMTensor = (*x).manager_ctx as *mut ArrayDLMTensor;
    let ctx: *mut NDArray = (*x).manager_ctx as *mut NDArray;
    std::mem::drop(ctx);
}

unsafe extern "C" fn destructor(o: *mut pyo3::ffi::PyObject) {
    println!("PyCapsule destructor");

    let name = CString::new("dltensor").unwrap();

    // let ptr = pyo3::ffi::PyCapsule_GetPointer(o, name.as_ptr()) as *mut dlpack::DLManagedTensor;

    let ptr = pyo3::ffi::PyCapsule_GetName(o);
    let current_name = CStr::from_ptr(ptr);
    println!("Current Name: {:?}", current_name);

    if current_name != name.as_c_str() {
        return;
    }

    let ptr = pyo3::ffi::PyCapsule_GetPointer(o, name.as_ptr()) as *mut dlpack::DLManagedTensor;
    (*ptr).deleter.unwrap()(ptr);

    println!("Delete by Python");

    // dbg!(*ptr);
    // (*ptr).deleter.unwrap()(ptr);
}

fn to_dlpack(src: Box<NDArray>) -> Box<dlpack::DLManagedTensor> {
    // let mut array_dlm_tensor = ArrayDLMTensor {

    //     tensor: dlpack::DLManagedTensor {
    //         dl_tensor: array_to_dl_tensor(src),
    //         manager_ctx: std::ptr::null_mut(),
    //         deleter: Some(deleter),
    //     },
    //     array: src,
    // };

    // array_dlm_tensor.tensor.manager_ctx =
    //     &mut array_dlm_tensor as *mut ArrayDLMTensor as *mut c_void;

    // let dlm_tensor = &mut array_dlm_tensor.tensor as *mut dlpack::DLManagedTensor;

    // std::mem::forget(array_dlm_tensor);

    let dl_tensor = array_to_dl_tensor(&src);
    // let ctx = Box::new(src);

    let dlm_tensor = dlpack::DLManagedTensor {
        dl_tensor,
        // manager_ctx: src as *mut NDArray as *mut c_void,
        manager_ctx: Box::into_raw(src) as *mut c_void,
        // manager_ctx: std::ptr::null_mut(),
        deleter: Some(deleter),
    };

    // let ptr = &mut dlm_tensor as *mut dlpack::DLManagedTensor;
    let ptr = Box::new(dlm_tensor);

    // std::mem::forget(dlm_tensor);

    ptr
}

#[pyfunction]
fn eye(n: usize) -> PyResult<*mut pyo3::ffi::PyObject> {
    let x: NDArray = ndarray::Array::eye(n).into_dyn();

    println!("eye = \n{}", x);
    let bx = Box::new(x);

    let dlm_tensor = to_dlpack(bx);
    let name = CString::new("dltensor").unwrap();

    let ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            Box::into_raw(dlm_tensor) as *mut c_void,
            name.as_ptr(),
            Some(destructor as pyo3::ffi::PyCapsule_Destructor),
        )
    };

    std::mem::forget(name);
    // std::mem::forget(dlm_tensor);
    // std::mem::forget(x);

    Ok(ptr)
}

/// A Python module implemented in Rust.
#[pymodule]
fn test_dlpack_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(eye))?;
    Ok(())
}
