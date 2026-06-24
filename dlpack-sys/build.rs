fn main() {
    let header = "vendor/dlpack/dlpack.h";
    println!("cargo:rerun-if-changed={header}");
    let bindings = bindgen::Builder::default()
        .header(header)
        .allowlist_type("DL.*")
        .allowlist_var("DLPACK_.*")
        .allowlist_var("kDL.*")
        .default_enum_style(bindgen::EnumVariation::ModuleConsts)
        .derive_copy(true)
        .derive_debug(true)
        .layout_tests(true)
        .generate()
        .expect("bindgen DLPack");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("write bindings");
}
