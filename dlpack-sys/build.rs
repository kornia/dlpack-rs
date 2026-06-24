fn main() {
    #[cfg(feature = "bindgen")]
    {
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
            .layout_tests(false) // committed output must be arch-portable (no baked size asserts)
            .generate()
            .expect("bindgen DLPack");
        bindings
            .write_to_file(std::path::Path::new("src").join("bindings.rs"))
            .expect("write src/bindings.rs");
    }
    // default: nothing — src/bindings.rs (committed) is used by lib.rs
}
