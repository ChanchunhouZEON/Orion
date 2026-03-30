use rand::RngExt;

/// Generate a random level for a new node using the HNSW formula:
/// l = floor(-ln(uniform(0,1)) * ml)
/// where ml = 1/ln(M).
pub fn random_level(ml: f64) -> usize {
    let mut rng = rand::rng();
    let r: f64 = rng.random_range(0f64..1f64);
    (-r.ln() * ml).floor() as usize
}
