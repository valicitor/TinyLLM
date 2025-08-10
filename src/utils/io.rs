use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};
use thiserror::Error;
use serde::{Serialize, de::DeserializeOwned};
use bincode;

/// Error type for I/O operations
#[derive(Error, Debug)]
pub enum IoError {
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),
    #[error("IO error: {0}")]
    StdIo(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    #[error("Invalid file format: {0}")]
    InvalidFormat(String),
}

/// Reads a file to string with proper error handling
pub fn read_to_string(path: impl AsRef<Path>) -> Result<String, IoError> {
    let path = path.as_ref();
    fs::read_to_string(path)
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                IoError::FileNotFound(path.to_path_buf())
            } else {
                IoError::StdIo(e)
            }
        })
}

/// Writes string to file, creating parent directories if needed
pub fn write_string(
    path: impl AsRef<Path>, 
    contents: impl AsRef<str>
) -> Result<(), IoError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents.as_ref())?;
    Ok(())
}

/// Serializes data to a binary file using bincode
pub fn serialize_to_file<T: Serialize>(
    path: impl AsRef<Path>,
    data: &T,
) -> Result<(), IoError> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, data)?;
    Ok(())
}

/// Deserializes data from a binary file using bincode
pub fn deserialize_from_file<T: DeserializeOwned>(
    path: impl AsRef<Path>,
) -> Result<T, IoError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}

/// Reads a file in chunks (useful for large files)
pub fn read_in_chunks(
    path: impl AsRef<Path>,
    chunk_size: usize,
    mut callback: impl FnMut(&[u8]) -> Result<(), IoError>,
) -> Result<(), IoError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buffer = vec![0; chunk_size];

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        callback(&buffer[..bytes_read])?;
    }

    Ok(())
}

/// Helper function to get all files with a specific extension in a directory
pub fn get_files_with_extension(
    dir: impl AsRef<Path>,
    extension: &str,
) -> Result<Vec<PathBuf>, IoError> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == extension {
                    files.push(path);
                }
            }
        }
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use serde::{Serialize, Deserialize};

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestData {
        value: i32,
        text: String,
    }

    #[test]
    fn test_serialize_deserialize() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.bin");

        let data = TestData {
            value: 42,
            text: "hello".to_string(),
        };

        serialize_to_file(&file_path, &data).unwrap();
        let loaded: TestData = deserialize_from_file(&file_path).unwrap();

        assert_eq!(data, loaded);
    }

    #[test]
    fn test_read_write_string() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        write_string(&file_path, "test content").unwrap();
        let content = read_to_string(&file_path).unwrap();

        assert_eq!(content, "test content");
    }
}