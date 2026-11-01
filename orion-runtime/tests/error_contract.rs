use orion_ann_runtime::{BaseInput, ErrorCode, Index, IndexOptions};

#[test]
fn manually_assembled_array_shapes_return_errors_before_slicing() {
    let mut input = BaseInput::from_f32(&[1.0; 24], 3).unwrap();
    if let BaseInput::F32 { dimension, .. } = &mut input {
        *dimension = 4;
    }
    let error = match Index::new(input, IndexOptions::default()) {
        Err(error) => error,
        Ok(_) => panic!("inconsistent array shape accepted"),
    };
    assert_eq!(error.code(), ErrorCode::DimensionMismatch);
    assert_eq!(error.details()["expected"], 3);
    assert_eq!(error.details()["actual"], 4);
}
