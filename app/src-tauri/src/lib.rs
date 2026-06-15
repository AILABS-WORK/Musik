use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

/// Holds the spawned Python sidecar so we can kill it on exit.
struct Sidecar(Mutex<Option<Child>>);

/// Locate the engine venv's Python. Order: $MGC_PYTHON, then walk up from the
/// executable dir and the cwd looking for `engine/.venv/.../python`.
fn find_python() -> Option<String> {
    if let Ok(p) = std::env::var("MGC_PYTHON") {
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
    }
    let rel = if cfg!(windows) {
        "engine/.venv/Scripts/python.exe"
    } else {
        "engine/.venv/bin/python"
    };
    let mut starts: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        starts.push(exe);
    }
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    for start in starts {
        let mut dir: &std::path::Path = start.as_path();
        loop {
            let cand = dir.join(rel);
            if cand.exists() {
                return cand.to_str().map(|s| s.to_string());
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => break,
            }
        }
    }
    None
}

/// Best-effort: start the FastAPI sidecar (uvicorn) on 127.0.0.1:8000.
/// If Python can't be found, we log and continue — the UI shows "sidecar
/// offline" and the user can run `mgc serve` manually.
fn spawn_sidecar() -> Option<Child> {
    let py = match find_python() {
        Some(p) => p,
        None => {
            log::warn!("mgc: engine venv python not found; run `mgc serve` manually");
            return None;
        }
    };
    let mut cmd = Command::new(&py);
    cmd.args([
        "-m", "uvicorn", "mgc.server:app",
        "--host", "127.0.0.1", "--port", "8000",
        "--log-level", "warning",
    ]);
    if let Ok(cfg) = std::env::var("MGC_CONFIG") {
        cmd.env("MGC_CONFIG", cfg);
    }
    match cmd.spawn() {
        Ok(child) => {
            log::info!("mgc sidecar started via {}", py);
            Some(child)
        }
        Err(e) => {
            log::error!("mgc: failed to start sidecar: {}", e);
            None
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            app.manage(Sidecar(Mutex::new(spawn_sidecar())));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(state) = app_handle.try_state::<Sidecar>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(child) = guard.as_mut() {
                            let _ = child.kill();
                        }
                    }
                }
            }
        });
}
