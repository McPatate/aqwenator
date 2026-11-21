//! Replay a trace of expert paging (`AQ_EXPERT_TRACE` of aq-engine) against a real disk, without
//! a model or a GPU: which records each layer needed and which were guessed for the layers
//! after it. The device and host tiers are simulated (LRU over records); the reads are real.
//! It measures how long a layer waits for the disk under a read policy and a read backend.
//!
//!     aq-replay <trace> [--dev-gib 3] [--ram-gib 4] [--policy sync|ahead] [--backend direct|uring|buffered|mmap]
//!               [--threads 8] [--layer-us 300] [--steps N] [--max-secs S] [--fixed] [--sqpoll]
//!     aq-replay <trace> --probe          random reads of one record at depths 1, 8, 32, 128
//!
//! Policies: `sync` reads a record when its layer needs it and waits. `ahead` also starts
//! reading the guesses for the next two layers and, at a layer, waits only for what that layer
//! needs. Backends: `direct` is O_DIRECT preads from a thread pool into aligned slots; `uring`
//! the same reads through one io_uring (Linux); `mmap` copies from a mapping of the files (the
//! page cache is then the host tier, sized by the machine or the container); `buffered` is
//! `direct` without O_DIRECT.
//!
//! The mapping can be helped (Linux): `--madvise random` turns the kernel's sequential
//! read-ahead off; `--mmap willneed` asks for a layer's records before touching them and
//! `--mmap populate` faults them in on the thread pool, whole records at a time; `--mmap-evict`
//! keeps the `--ram-gib` most recently used records and tells the kernel to drop the others;
//! with the `ahead` policy the guesses are asked for with MADV_WILLNEED.
//! `--layer-us` is the compute time of a layer, spent between its paging calls. With `uring`,
//! `--fixed` registers the host slots with the ring once (no page pinning per read) and
//! `--sqpoll` lets a kernel thread take the submissions.

use anyhow::{Context, Result, bail};
use aq_io::{ALIGN, Io, Read, aligned};
use std::collections::HashMap;
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

const NONE: u32 = u32::MAX;

struct Layout {
    rec: usize,
    per_layer: usize,
    /// Per layer: index into `files` and the offset of its first record.
    layers: Vec<(usize, u64)>,
    files: Vec<std::fs::File>,
    maps: Vec<(usize, usize)>,
}

impl Layout {
    fn open(path: &str, direct: bool) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        let mut lines = text.lines();
        let mut head = lines.next().context("empty layout")?.split_whitespace().map(|v| v.parse::<usize>());
        let (rec, per_layer) = (head.next().context("layout header")??, head.next().context("layout header")??);
        let (mut files, mut layers, mut seen) = (Vec::new(), Vec::new(), HashMap::new());
        for l in lines.filter(|l| !l.is_empty()) {
            let (p, at) = l.rsplit_once(' ').context("layout line")?;
            let fi = match seen.get(p) {
                Some(&fi) => fi,
                None => {
                    let mut o = std::fs::OpenOptions::new();
                    o.read(true);
                    #[cfg(target_os = "linux")]
                    if direct {
                        use std::os::unix::fs::OpenOptionsExt;
                        o.custom_flags(libc::O_DIRECT);
                    }
                    let f = o.open(p).with_context(|| format!("open {p}"))?;
                    #[cfg(target_os = "macos")]
                    if direct {
                        unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1) };
                    }
                    files.push(f);
                    seen.insert(p.to_string(), files.len() - 1);
                    files.len() - 1
                }
            };
            layers.push((fi, at.parse()?));
        }
        Ok(Self { rec, per_layer, layers, files, maps: Vec::new() })
    }

    fn at(&self, id: u32) -> (usize, u64) {
        let (fi, first) = self.layers[id as usize / self.per_layer];
        (fi, first + ((id as usize % self.per_layer) * self.rec) as u64)
    }
}

/// Slots holding the least recently used of a larger set of records.
struct Lru {
    owner: Vec<u32>,
    stamp: Vec<u64>,
    slot: Vec<u32>,
}

impl Lru {
    fn new(slots: usize, records: usize) -> Self {
        Self { owner: vec![NONE; slots], stamp: vec![0; slots], slot: vec![NONE; records] }
    }

    fn has(&self, id: u32) -> bool {
        self.slot[id as usize] != NONE
    }

    fn touch(&mut self, id: u32, tick: u64) {
        let s = self.slot[id as usize];
        if s != NONE && self.stamp[s as usize] != u64::MAX {
            self.stamp[s as usize] = tick;
        }
    }

    /// A slot for `id`, taken from the least recently used record not used at `tick` (and not
    /// pinned with `u64::MAX`); the record it replaces.
    fn place(&mut self, id: u32, tick: u64) -> Option<(u32, u32)> {
        let s = (0..self.owner.len() as u32).filter(|&s| self.stamp[s as usize] < tick).min_by_key(|&s| self.stamp[s as usize])?;
        let old = self.owner[s as usize];
        if old != NONE {
            self.slot[old as usize] = NONE;
        }
        self.owner[s as usize] = id;
        self.slot[id as usize] = s;
        self.stamp[s as usize] = tick;
        Some((s, old))
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * p) as usize]
}

fn spin(d: Duration) {
    let t = Instant::now();
    while t.elapsed() < d {
        std::hint::spin_loop();
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |name: &str, default: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or(default.to_string());
    let Some(trace) = args.first().filter(|a| !a.starts_with("--")) else { bail!("usage: aq-replay <trace> [options]; see the top of main.rs") };
    let (policy, backend) = (opt("--policy", "sync"), opt("--backend", "direct"));
    let (dev_gib, ram_gib): (f64, f64) = (opt("--dev-gib", "3").parse()?, opt("--ram-gib", "4").parse()?);
    let max_secs: u64 = opt("--max-secs", "0").parse()?;
    let (threads, layer_us, max_steps): (usize, u64, usize) = (opt("--threads", "8").parse()?, opt("--layer-us", "300").parse()?, opt("--steps", "0").parse()?);
    let mut lay = Layout::open(&opt("--layout", &format!("{trace}.layout")), backend != "mmap" && backend != "buffered")?;
    let (rec, n) = (lay.rec, lay.layers.len() * lay.per_layer);
    let stride = rec.next_multiple_of(ALIGN) + ALIGN;
    let flag = |name: &str| args.iter().any(|a| a == name);
    let mut io = if backend == "uring" { Io::uring(flag("--sqpoll"))? } else { Io::threads(threads) };
    let (mm_mode, mm_evict, mm_random) = (opt("--mmap", "fault"), flag("--mmap-evict"), opt("--madvise", "normal") == "random");
    let submit = |io: &mut Io, id: u32, buf: usize, lay: &Layout| {
        let (fi, at) = lay.at(id);
        let head = (at % ALIGN as u64) as usize;
        io.submit(Read { id, fd: lay.files[fi].as_raw_fd(), at: at - head as u64, buf, len: (head + rec).next_multiple_of(ALIGN) });
    };

    if args.iter().any(|a| a == "--probe") {
        let region = aligned(128 * stride);
        if flag("--fixed") {
            io.register(region, 128 * stride, stride)?;
        }
        let bufs: Vec<usize> = (0..128).map(|i| region + i * stride).collect();
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as u32
        };
        println!("{n} records of {:.2} MiB in {} files; random direct reads, {}", rec as f64 / 1048576.0, lay.files.len(), if backend == "uring" { format!("io_uring{}{}", if flag("--fixed") { ", fixed buffers" } else { "" }, if flag("--sqpoll") { ", sqpoll" } else { "" }) } else { format!("{threads} threads") });
        for depth in [1usize, 8, 32, 128] {
            let (t, mut lat) = (Instant::now(), Vec::new());
            let mut count = 0;
            for _ in 0..(256 / depth).max(8) {
                let t0 = Instant::now();
                for b in &bufs[..depth] {
                    submit(&mut io, next(), *b, &lay);
                }
                io.flush();
                for _ in 0..depth {
                    io.wait_done();
                    count += 1;
                }
                lat.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            let s = t.elapsed().as_secs_f64();
            println!("depth {depth:3}: a batch takes {:.2} ms (p50, p99 {:.2}), {:.0} reads/s, {:.2} GiB/s", pct(&mut lat, 0.5), pct(&mut lat, 0.99), count as f64 / s, (count * rec) as f64 / s / (1u64 << 30) as f64);
        }
        return Ok(());
    }

    let raw = std::fs::read(trace)?;
    let words: Vec<u32> = raw.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let dev_cap = ((dev_gib * (1u64 << 30) as f64) as usize / rec).min(n);
    let ram_cap = ((ram_gib * (1u64 << 30) as f64) as usize / stride).min(n);
    let (mut dev, mut ram) = (Lru::new(dev_cap, n), Lru::new(ram_cap, n));
    let mmap = backend == "mmap";
    let host = if mmap { 0 } else { aligned(ram_cap * stride) };
    if flag("--fixed") && !mmap {
        io.register(host, ram_cap * stride, stride)?;
    }
    if mmap {
        for f in &lay.files {
            let len = f.metadata()?.len() as usize;
            // Start cold where the system lets a process drop a file's cached pages.
            #[cfg(target_os = "linux")]
            unsafe {
                libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED)
            };
            let p = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, f.as_raw_fd(), 0) };
            anyhow::ensure!(p != libc::MAP_FAILED, "mmap failed");
            if mm_random {
                unsafe { libc::madvise(p, len, libc::MADV_RANDOM) };
            }
            lay.maps.push((p as usize, len));
        }
    }
    let scratch = aligned(stride);
    // A record's pages in the mapping (those it has alone when dropping it).
    let pages = |lay: &Layout, id: u32, inner: bool| {
        let (fi, at) = lay.at(id);
        let (a, b) = (lay.maps[fi].0 + at as usize, lay.maps[fi].0 + at as usize + rec);
        if inner { (a.next_multiple_of(ALIGN), b / ALIGN * ALIGN) } else { (a / ALIGN * ALIGN, b.next_multiple_of(ALIGN)) }
    };
    let advise = |lay: &Layout, id: u32, advice: i32, inner: bool| {
        let (a, b) = pages(lay, id, inner);
        unsafe { libc::madvise(a as *mut libc::c_void, b - a, advice) };
    };
    #[cfg(target_os = "linux")]
    let drop_advice = libc::MADV_PAGEOUT;
    #[cfg(not(target_os = "linux"))]
    let drop_advice = libc::MADV_DONTNEED;
    let mut mm_need: Vec<u32> = Vec::new();
    let mut reading: HashMap<u32, ()> = HashMap::new();
    let (mut tick, mut i, mut step_wait, mut step_layers) = (0u64, 0usize, 0.0f64, 0usize);
    let (mut steps, mut layer_wait, mut prev_first) = (Vec::new(), Vec::new(), u32::MAX);
    let (mut needed, mut dev_miss, mut disk_needed, mut disk_ahead, mut ahead_used, mut rows_total) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut was_ahead = vec![false; n];
    let t_all = Instant::now();
    while i + 4 <= words.len() {
        let (rows, counts) = (words[i] as usize, [words[i + 1] as usize, words[i + 2] as usize, words[i + 3] as usize]);
        let need = &words[i + 4..i + 4 + counts[0]];
        let next = &words[i + 4 + counts[0]..i + 4 + counts[0] + counts[1]];
        let far = &words[i + 4 + counts[0] + counts[1]..i + 4 + counts[0] + counts[1] + counts[2]];
        i += 4 + counts.iter().sum::<usize>();
        let decode = rows <= 64;
        // A forward starts when the layers start over.
        if need.first().is_some_and(|&f| f <= prev_first) && step_layers > 0 {
            steps.push(step_wait);
            (step_wait, step_layers) = (0.0, 0);
            if max_steps > 0 && steps.len() >= max_steps || max_secs > 0 && t_all.elapsed().as_secs() >= max_secs {
                break;
            }
        }
        prev_first = need.first().copied().unwrap_or(prev_first);
        tick += 1;
        let t = Instant::now();
        // Completions that arrived meanwhile.
        while let Some(id) = io.try_done() {
            reading.remove(&id);
            ram.stamp[ram.slot[id as usize] as usize] = tick - 1;
        }
        let mut wait_for = Vec::new();
        for &id in need {
            needed += decode as usize;
            if dev.has(id) {
                dev.touch(id, tick);
                ram.touch(id, tick);
                continue;
            }
            dev_miss += decode as usize;
            if mmap {
                mm_need.push(id);
            } else if reading.contains_key(&id) {
                wait_for.push(id);
            } else if ram.has(id) {
                ahead_used += (decode && std::mem::take(&mut was_ahead[id as usize])) as usize;
                ram.touch(id, tick);
            } else if let Some((s, _)) = ram.place(id, tick) {
                disk_needed += decode as usize;
                ram.stamp[s as usize] = u64::MAX;
                reading.insert(id, ());
                submit(&mut io, id, host + s as usize * stride, &lay);
                wait_for.push(id);
            }
            dev.place(id, tick);
        }
        if mmap {
            for &id in &mm_need {
                if mm_evict {
                    if ram.has(id) {
                        ram.touch(id, tick);
                    } else if let Some((_, old)) = ram.place(id, tick) {
                        disk_needed += decode as usize;
                        if old != NONE {
                            advise(&lay, old, drop_advice, true);
                        }
                    }
                }
                match mm_mode.as_str() {
                    "willneed" => advise(&lay, id, libc::MADV_WILLNEED, false),
                    "populate" => {
                        let (a, b) = pages(&lay, id, false);
                        io.submit(Read { id, fd: -1, at: 0, buf: a, len: b - a });
                    }
                    _ => {}
                }
            }
            if mm_mode == "populate" {
                for _ in &mm_need {
                    io.wait_done();
                }
            }
            for id in mm_need.drain(..) {
                let (fi, at) = lay.at(id);
                unsafe { std::ptr::copy_nonoverlapping((lay.maps[fi].0 + at as usize) as *const u8, scratch as *mut u8, rec) };
            }
        }
        io.flush();
        while wait_for.iter().any(|id| reading.contains_key(id)) {
            let id = io.wait_done();
            reading.remove(&id);
            ram.stamp[ram.slot[id as usize] as usize] = tick;
        }
        for id in wait_for {
            ahead_used += (decode && std::mem::take(&mut was_ahead[id as usize])) as usize;
        }
        // Guesses: what host memory holds goes to the device; with `ahead`, the rest is read.
        for (dist, guess) in [next, far].into_iter().enumerate() {
            for &id in guess {
                if dev.has(id) {
                    dev.touch(id, tick);
                } else if mmap {
                    // The kernel is asked for the guess; with our own eviction it also takes a place.
                    if policy == "ahead" && !(mm_evict && ram.has(id)) {
                        disk_ahead += decode as usize;
                        advise(&lay, id, libc::MADV_WILLNEED, false);
                        if mm_evict {
                            if let Some((_, old)) = ram.place(id, tick) {
                                if old != NONE {
                                    advise(&lay, old, drop_advice, true);
                                }
                            }
                        }
                    }
                } else if !mmap && ram.has(id) && !reading.contains_key(&id) {
                    if dist == 0 {
                        dev.place(id, tick);
                    }
                } else if !mmap && policy == "ahead" && !reading.contains_key(&id) {
                    if let Some((s, _)) = ram.place(id, tick) {
                        disk_ahead += decode as usize;
                        was_ahead[id as usize] = true;
                        ram.stamp[s as usize] = u64::MAX;
                        reading.insert(id, ());
                        submit(&mut io, id, host + s as usize * stride, &lay);
                    }
                }
            }
        }
        io.flush();
        let w = t.elapsed().as_secs_f64() * 1e3;
        if decode {
            layer_wait.push(w);
            step_wait += w;
            step_layers += 1;
            rows_total += rows;
        }
        spin(Duration::from_micros(layer_us));
    }
    let layers_per_step = layer_wait.len() as f64 / steps.len().max(1) as f64;
    let mean = steps.iter().sum::<f64>() / steps.len().max(1) as f64;
    let mib = |c: usize| (c * rec) as f64 / steps.len().max(1) as f64 / 1048576.0;
    let backend = format!("{backend}{}{}{}", if flag("--fixed") { "+fixed" } else { "" }, if flag("--sqpoll") { "+sqpoll" } else { "" }, if mmap { format!(" ({mm_mode}{}{})", if mm_random { ", random" } else { "" }, if mm_evict { ", own eviction" } else { "" }) } else { String::new() });
    println!("{policy}/{backend}: device {dev_gib} GiB ({dev_cap} records), host {} ({ram_cap}), {layer_us} us per layer; {} forwards in {:.1} s", if mmap && !mm_evict { "page cache".into() } else { format!("{ram_gib} GiB") }, steps.len(), t_all.elapsed().as_secs_f64());
    println!("  per forward: {:.0} records needed, {:.1}% not on the device, {:.0} MiB read when needed, {:.0} MiB read ahead ({:.0}% of it used)", needed as f64 / steps.len().max(1) as f64, 100.0 * dev_miss as f64 / needed.max(1) as f64, mib(disk_needed), mib(disk_ahead), 100.0 * ahead_used as f64 / disk_ahead.max(1) as f64);
    println!("  wait per forward: mean {mean:.2} ms, p50 {:.2}, p99 {:.2}; per layer: p50 {:.3}, p99 {:.3}", pct(&mut steps, 0.5), pct(&mut steps, 0.99), pct(&mut layer_wait, 0.5), pct(&mut layer_wait, 0.99));
    println!("  forwards per second at this compute time: {:.1} ({:.1} rows per forward)", 1e3 / (mean + layers_per_step * layer_us as f64 / 1e3), rows_total as f64 / layer_wait.len().max(1) as f64);
    Ok(())
}
