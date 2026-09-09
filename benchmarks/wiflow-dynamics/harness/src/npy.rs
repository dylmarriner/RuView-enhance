//! Minimal read-only `.npy` reader over a memory map.
//!
//! Deliberately minimal: it supports exactly the dtypes this benchmark uses and
//! refuses anything else loudly rather than guessing. `csi_windows.npy` is 15.5 GB,
//! so mapping rather than reading is a requirement, not an optimisation.

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    I32,
    I64,
    Bool,
}

impl Dtype {
    fn parse(descr: &str) -> Result<Self, String> {
        match descr {
            "<f4" | "=f4" => Ok(Dtype::F32),
            "<i4" | "=i4" => Ok(Dtype::I32),
            "<i8" | "=i8" => Ok(Dtype::I64),
            "|b1" => Ok(Dtype::Bool),
            other => Err(format!(
                "unsupported .npy dtype {other:?} (this reader handles <f4, <i4, <i8, |b1)"
            )),
        }
    }

    fn size(self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::I64 => 8,
            Dtype::Bool => 1,
        }
    }
}

pub struct Npy {
    mmap: Mmap,
    data_offset: usize,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

/// Extract `key`'s value from a numpy header dict without a full Python parser.
fn header_value<'a>(header: &'a str, key: &str) -> Result<&'a str, String> {
    let pat = format!("'{key}':");
    let start = header
        .find(&pat)
        .ok_or_else(|| format!("missing {key:?} in .npy header"))?
        + pat.len();
    let rest = &header[start..];
    // Value ends at the comma that is not inside a tuple.
    let mut depth = 0usize;
    for (i, c) in rest.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => return Ok(rest[..i].trim()),
            _ => {}
        }
    }
    Ok(rest.trim().trim_end_matches('}').trim())
}

impl Npy {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        // SAFETY: the fixture and dataset files are read-only inputs; we never write
        // through this map and the process does not mutate them concurrently.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", path.display()))?;

        if mmap.len() < 10 || &mmap[0..6] != b"\x93NUMPY" {
            return Err(format!("{} is not a .npy file", path.display()));
        }
        let major = mmap[6];
        let (header_len, header_start) = match major {
            1 => (u16::from_le_bytes([mmap[8], mmap[9]]) as usize, 10),
            2 | 3 => (
                u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize,
                12,
            ),
            v => return Err(format!("unsupported .npy major version {v}")),
        };
        let header = std::str::from_utf8(&mmap[header_start..header_start + header_len])
            .map_err(|e| format!("non-utf8 .npy header: {e}"))?;

        let dtype = Dtype::parse(header_value(header, "descr")?.trim_matches('\''))?;

        if header_value(header, "fortran_order")? != "False" {
            return Err("fortran-ordered .npy is not supported".into());
        }

        let shape_str = header_value(header, "shape")?;
        let shape: Vec<usize> = shape_str
            .trim_matches(|c| c == '(' || c == ')')
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>().map_err(|e| format!("bad shape {s:?}: {e}")))
            .collect::<Result<_, _>>()?;

        let data_offset = header_start + header_len;
        let expected = shape.iter().product::<usize>() * dtype.size();
        let actual = mmap.len() - data_offset;
        if actual < expected {
            return Err(format!(
                "{}: truncated -- header declares {expected} data bytes, file has {actual}",
                path.display()
            ));
        }
        Ok(Npy { mmap, data_offset, shape, dtype })
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn check(&self, want: Dtype) -> Result<(), String> {
        if self.dtype != want {
            return Err(format!("dtype mismatch: file is {:?}, requested {want:?}", self.dtype));
        }
        Ok(())
    }

    /// Copy the whole array out as f32. Callers that must not allocate should
    /// use [`Npy::f32_slice`] instead.
    pub fn to_f32(&self) -> Result<Vec<f32>, String> {
        Ok(self.f32_slice()?.to_vec())
    }

    /// Zero-copy view. Falls back to an error rather than a misaligned read.
    pub fn f32_slice(&self) -> Result<&[f32], String> {
        self.check(Dtype::F32)?;
        let bytes = &self.mmap[self.data_offset..self.data_offset + self.len() * 4];
        let (head, body, _) = unsafe { bytes.align_to::<f32>() };
        if !head.is_empty() || body.len() != self.len() {
            return Err("mmap data is not f32-aligned".into());
        }
        Ok(body)
    }

    pub fn to_i64(&self) -> Result<Vec<i64>, String> {
        let n = self.len();
        let base = self.data_offset;
        match self.dtype {
            Dtype::I64 => Ok((0..n)
                .map(|i| {
                    let o = base + i * 8;
                    i64::from_le_bytes(self.mmap[o..o + 8].try_into().unwrap())
                })
                .collect()),
            Dtype::I32 => Ok((0..n)
                .map(|i| {
                    let o = base + i * 4;
                    i32::from_le_bytes(self.mmap[o..o + 4].try_into().unwrap()) as i64
                })
                .collect()),
            d => Err(format!("dtype {d:?} is not an integer type")),
        }
    }

    pub fn to_bool(&self) -> Result<Vec<bool>, String> {
        self.check(Dtype::Bool)?;
        Ok(self.mmap[self.data_offset..self.data_offset + self.len()]
            .iter()
            .map(|&b| b != 0)
            .collect())
    }
}
