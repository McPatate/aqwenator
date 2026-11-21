//! Reads of byte ranges of files into caller-owned memory, queued and reaped one by one: a pool
//! of threads doing preads, or an io_uring (Linux). It knows nothing of what the bytes are.
//! For direct I/O the file offset, the length and the buffer are multiples of [`ALIGN`].

#[cfg(target_os = "linux")]
use anyhow::Context;
use anyhow::Result;
use std::sync::{Arc, Mutex, mpsc};

pub const ALIGN: usize = 4096;

pub struct Read {
    /// Returned when the read is done.
    pub id: u32,
    /// The file, or negative to fault in the mapped pages at `buf` instead (Linux).
    pub fd: i32,
    pub at: u64,
    pub buf: usize,
    pub len: usize,
}

/// Where the reads go: a pool of threads doing preads, or an io_uring.
pub enum Io {
    Threads { jobs: Option<mpsc::Sender<Read>>, done: mpsc::Receiver<u32>, threads: Vec<std::thread::JoinHandle<()>> },
    #[cfg(target_os = "linux")]
    Uring { ring: io_uring::IoUring, fixed: Option<(usize, usize, usize)> },
}

impl Io {
    /// One io_uring; with `sqpoll` a kernel thread takes the submissions.
    pub fn uring(sqpoll: bool) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            let mut b = io_uring::IoUring::builder();
            if sqpoll {
                b.setup_sqpoll(2000);
            }
            Ok(Io::Uring { ring: b.build(256)?, fixed: None })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = sqpoll;
            anyhow::bail!("io_uring needs Linux")
        }
    }

    pub fn threads(threads: usize) -> Self {
        let (jobs, rx) = mpsc::channel::<Read>();
        let rx = Arc::new(Mutex::new(rx));
        let (tx, done) = mpsc::channel();
        let threads = (0..threads)
            .map(|_| {
                let (rx, tx) = (rx.clone(), tx.clone());
                std::thread::spawn(move || {
                    loop {
                        let Ok(r) = rx.lock().unwrap().recv() else { break };
                        let mut got = 0;
                        #[cfg(target_os = "linux")]
                        if r.fd < 0 {
                            unsafe { libc::madvise(r.buf as *mut libc::c_void, r.len, libc::MADV_POPULATE_READ) };
                            got = r.len;
                        }
                        while got < r.len {
                            let n = unsafe { libc::pread(r.fd, (r.buf + got) as *mut libc::c_void, r.len - got, (r.at + got as u64) as libc::off_t) };
                            if n <= 0 {
                                break;
                            }
                            got += n as usize;
                        }
                        if tx.send(r.id).is_err() {
                            break;
                        }
                    }
                })
            })
            .collect();
        Io::Threads { jobs: Some(jobs), done, threads }
    }

    /// Register `bytes` at `base` as fixed buffers, in pieces of whole slots of at most 1 GiB.
    pub fn register(&mut self, base: usize, bytes: usize, stride: usize) -> Result<()> {
        #[cfg(target_os = "linux")]
        if let Io::Uring { ring, fixed } = self {
            let piece = ((1usize << 30) / stride).max(1) * stride;
            let iov: Vec<libc::iovec> = (0..bytes.div_ceil(piece)).map(|i| libc::iovec { iov_base: (base + i * piece) as *mut libc::c_void, iov_len: piece.min(bytes - i * piece) }).collect();
            unsafe { ring.submitter().register_buffers(&iov) }.context("registering buffers (the locked-memory limit may be too low)")?;
            *fixed = Some((base, bytes, piece));
        }
        let _ = (base, bytes, stride);
        Ok(())
    }

    /// Queue a read; `flush` hands the queued ones to the kernel.
    pub fn submit(&mut self, r: Read) {
        match self {
            Io::Threads { jobs, .. } => jobs.as_ref().unwrap().send(r).unwrap(),
            #[cfg(target_os = "linux")]
            Io::Uring { ring, fixed } => {
                let fd = io_uring::types::Fd(r.fd);
                let e = match fixed.filter(|f| r.buf >= f.0 && r.buf < f.0 + f.1) {
                    Some((base, _, piece)) => io_uring::opcode::ReadFixed::new(fd, r.buf as *mut u8, r.len as u32, ((r.buf - base) / piece) as u16).offset(r.at).build(),
                    None => io_uring::opcode::Read::new(fd, r.buf as *mut u8, r.len as u32).offset(r.at).build(),
                }
                .user_data(r.id as u64);
                while unsafe { ring.submission().push(&e) }.is_err() {
                    ring.submit().expect("io_uring submit");
                }
            }
        }
    }

    pub fn flush(&mut self) {
        #[cfg(target_os = "linux")]
        if let Io::Uring { ring, .. } = self {
            ring.submit().expect("io_uring submit");
        }
    }

    /// A finished read, if one is.
    pub fn try_done(&mut self) -> Option<u32> {
        match self {
            Io::Threads { done, .. } => done.try_recv().ok(),
            #[cfg(target_os = "linux")]
            Io::Uring { ring, .. } => ring.completion().next().map(|c| {
                assert!(c.result() > 0, "read failed: {}", c.result());
                c.user_data() as u32
            }),
        }
    }

    /// The next finished read, waiting for it.
    pub fn wait_done(&mut self) -> u32 {
        loop {
            if let Some(id) = self.try_done() {
                return id;
            }
            match self {
                Io::Threads { done, .. } => return done.recv().expect("readers gone"),
                #[cfg(target_os = "linux")]
                Io::Uring { ring, .. } => {
                    ring.submit_and_wait(1).expect("io_uring wait");
                }
            }
        }
    }
}

impl Drop for Io {
    fn drop(&mut self) {
        #[allow(irrefutable_let_patterns)]
        if let Io::Threads { jobs, threads, .. } = self {
            *jobs = None;
            for t in threads.drain(..) {
                let _ = t.join();
            }
        }
    }
}

/// `bytes` of memory aligned for direct reads; never freed.
pub fn aligned(bytes: usize) -> usize {
    let mut p = std::ptr::null_mut();
    assert_eq!(unsafe { libc::posix_memalign(&mut p, ALIGN, bytes) }, 0, "out of memory");
    p as usize
}
