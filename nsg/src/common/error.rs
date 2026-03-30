use thiserror::Error;

#[derive(Error, Debug)]
pub enum NSGError {
    #[error("Index error: {0}")]
    IndexError(String),

    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type NSGResult<T> = Result<T, NSGError>;
