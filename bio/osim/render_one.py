#!/usr/bin/env python3
"""Single-process render: vectorized RenderCtx -> raw frames piped straight into ffmpeg
(no PNG, no disk round-trip). Reports setup / render / per-frame timing.
Args: sto out_mp4 [t0 t1]   Env: RENDER_FPS, RENDER_STEP, COLOR_GAIN"""
import os, sys, time, subprocess
import render_core as rc


def main():
    sto = sys.argv[1]; out = sys.argv[2]
    t0 = float(sys.argv[3]) if len(sys.argv) > 3 else None
    t1 = float(sys.argv[4]) if len(sys.argv) > 4 else None
    fps = int(os.environ.get("RENDER_FPS", "50"))
    step = int(os.environ.get("RENDER_STEP", "1"))
    gain = float(os.environ.get("COLOR_GAIN", "1.0"))

    t = time.time()
    ctx = rc.RenderCtx(sto, gain)
    fr = os.environ.get("RENDER_FRAMES")          # "a:b" exact frame-index slice (parallel chunks)
    if fr:
        a, b = (int(x) if x else None for x in fr.split(":"))
        frames = ctx.frames_in_range(None, None, step)[a:b]
    else:
        frames = ctx.frames_in_range(t0, t1, step)
    setup_s = time.time() - t

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{rc.W}x{rc.H}",
         "-framerate", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t = time.time(); n = 0
    for fi in frames:
        img = ctx.render_frame(int(fi))
        if img is None:
            continue
        ff.stdin.write(img.tobytes()); n += 1
    ff.stdin.close(); ff.wait()
    render_s = time.time() - t
    ctx.close()
    print(f"RENDER_ONE frames={n} setup={setup_s:.1f}s render={render_s:.1f}s "
          f"per_frame={render_s/max(n,1)*1000:.0f}ms fps={n/max(render_s,1e-9):.1f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
