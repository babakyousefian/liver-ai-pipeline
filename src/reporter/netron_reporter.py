# src/reporter/netron_reporter.py

"""
Netron Reporter — ONNX Export + HTML Visualization

✅ FIX 1: ONNX export حالا device mismatch را handle می‌کند
✅ FIX 2: HTML visualization با Netron CDN + fallback محلی ساخته می‌شود
✅ ZERO impact on model performance — only runs after training
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def save_model_with_name(
    model: nn.Module,
    save_dir: str,
    model_name: str = "model",
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
) -> Dict[str, Optional[str]]:
    """
    Export model to ONNX and generate standalone HTML visualization.

    Args:
        model: PyTorch model to export
        save_dir: Directory to save files
        model_name: Name of the model (used in filenames)
        input_shape: Input tensor shape (batch, channels, height, width)

    Returns:
        dict with keys: onnx_path, html_path, info_path, success
    """
    os.makedirs(save_dir, exist_ok=True)

    onnx_path = os.path.join(save_dir, f"{model_name}.onnx")
    html_path = os.path.join(save_dir, f"{model_name}.html")
    info_path = os.path.join(save_dir, f"{model_name}_architecture_info.json")

    # ============================================================
    # ✅ FIX 1: تشخیص device مدل و ساخت dummy input روی همان device
    # ============================================================
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    print(f"📌 Model device: {model_device}")

    # ============================================================
    # ✅ ONNX EXPORT
    # ============================================================
    onnx_ok = False
    onnx_error = None

    try:
        # ✅ اطمینان از eval mode (بدون تغییر در وزن‌ها)
        was_training = model.training
        model.eval()

        # ✅ dummy input روی همان device مدل
        dummy_input = torch.randn(*input_shape, device=model_device)

        # ✅ export با تنظیمات مناسب برای ResNet/EfficientNet
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=13,               # opset 13 = سازگاری گسترده
                do_constant_folding=True,       # بهینه‌سازی گراف
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
                verbose=False,
            )

        # ✅ بازگشت به حالت اولیه (اگر train بود)
        if was_training:
            model.train()

        onnx_ok = os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0

        if onnx_ok:
            size_mb = os.path.getsize(onnx_path) / 1e6
            print(f"✅ ONNX exported: {onnx_path} ({size_mb:.2f} MB)")
        else:
            print(f"⚠️ ONNX file empty or missing")
            onnx_path = None

    except Exception as e:
        onnx_error = str(e)
        print(f"⚠️ ONNX export failed: {e}")
        onnx_path = None
        # ✅ cleanup ناقص‌ها
        if os.path.exists(onnx_path) if onnx_path else False:
            try:
                os.remove(onnx_path)
            except Exception:
                pass

    # ============================================================
    # ✅ HTML GENERATION
    # ============================================================
    html_ok = False

    if onnx_ok and onnx_path:
        try:
            _generate_netron_html(
                html_path=html_path,
                onnx_filename=os.path.basename(onnx_path),
                model_name=model_name,
                num_params=sum(p.numel() for p in model.parameters()),
            )
            html_ok = os.path.exists(html_path) and os.path.getsize(html_path) > 0

            if html_ok:
                print(f"✅ HTML generated: {html_path}")
            else:
                print(f"⚠️ HTML file empty or missing")
                html_path = None

        except Exception as e:
            print(f"⚠️ HTML generation failed: {e}")
            html_path = None
    else:
        html_path = None

    # ============================================================
    # ✅ ذخیره‌ی info JSON
    # ============================================================
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "model_name": model_name,
        "input_shape": list(input_shape),
        "num_parameters": num_params,
        "num_trainable_parameters": num_trainable,
        "exported_at": datetime.now().isoformat(),
        "onnx_path": onnx_path,
        "html_path": html_path,
        "device": str(model_device),
        "onnx_export_success": onnx_ok,
        "html_generation_success": html_ok,
    }

    if onnx_error:
        info["onnx_error"] = onnx_error

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    print(f"✅ Model info saved: {info_path}")

    return {
        "onnx_path": onnx_path,
        "html_path": html_path,
        "info_path": info_path,
        "success": onnx_ok and html_ok,
    }


def _generate_netron_html(
    html_path: str,
    onnx_filename: str,
    model_name: str,
    num_params: int,
) -> None:
    """
    Generate standalone HTML file that displays the ONNX model using Netron.

    - Primary: Netron via CDN (needs internet)
    - Fallback: local summary table with parameter counts
    """
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Architecture — {model_name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: 100%; height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1e1e1e; color: #d4d4d4; overflow: hidden; }}
  #header {{ height: 56px; background: #252526; border-bottom: 1px solid #3e3e42; display: flex; align-items: center; padding: 0 20px; justify-content: space-between; }}
  #header .title {{ font-size: 16px; font-weight: 600; color: #ffffff; }}
  #header .badge {{ padding: 3px 10px; background: #0e639c; color: white; border-radius: 4px; font-size: 12px; margin-left: 12px; }}
  #header .stats {{ font-size: 12px; color: #858585; }}
  #viewer {{ width: 100%; height: calc(100% - 56px); }}
  #fallback {{ display: none; padding: 40px; max-width: 900px; margin: 0 auto; overflow-y: auto; height: calc(100% - 56px); }}
  #fallback h2 {{ color: #f48771; margin-bottom: 20px; }}
  #fallback p {{ margin-bottom: 15px; line-height: 1.6; color: #cccccc; }}
  #fallback table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #252526; border-radius: 6px; overflow: hidden; }}
  #fallback th, #fallback td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #3e3e42; }}
  #fallback th {{ background: #2d2d30; color: #ffffff; font-weight: 600; }}
  #fallback code {{ background: #2d2d30; padding: 2px 8px; border-radius: 3px; font-family: 'Courier New', monospace; color: #ce9178; }}
  #fallback a {{ color: #4fc3f7; text-decoration: none; }}
  #fallback a:hover {{ text-decoration: underline; }}
  #loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #858585; }}
</style>
</head>
<body>
  <div id="header">
    <div>
      <span class="title">{model_name}</span>
      <span class="badge">ONNX</span>
    </div>
    <div class="stats">{num_params:,} parameters</div>
  </div>

  <div id="viewer">
    <div id="loading">⏳ Loading Netron viewer...</div>
  </div>

  <div id="fallback">
    <h2>⚠️ Netron Viewer Not Available</h2>
    <p>
      The Netron library could not be loaded from CDN. This usually happens when
      there is no internet connection.
    </p>
    <p>
      <strong>Solution 1:</strong> Check your internet connection and reload this page.
    </p>
    <p>
      <strong>Solution 2:</strong> Open the ONNX file
      <code>{onnx_filename}</code> with the Netron desktop app.
      Download from <a href="https://github.com/lutzroeder/netron/releases" target="_blank">
      https://github.com/lutzroeder/netron/releases</a>.
    </p>
    <p>
      <strong>Solution 3:</strong> Upload <code>{onnx_filename}</code> to
      <a href="https://netron.app" target="_blank">https://netron.app</a>.
    </p>

    <h3 style="margin-top:30px; color:#fff;">Model Summary</h3>
    <table>
      <tr><th>Property</th><th>Value</th></tr>
      <tr><td>Model Name</td><td>{model_name}</td></tr>
      <tr><td>Format</td><td>ONNX</td></tr>
      <tr><td>Parameters</td><td>{num_params:,}</td></tr>
      <tr><td>ONNX File</td><td><code>{onnx_filename}</code></td></tr>
    </table>
  </div>

  <script>
    (function() {{
      var loaded = false;

      function showFallback() {{
        if (loaded) return;
        document.getElementById('viewer').style.display = 'none';
        document.getElementById('fallback').style.display = 'block';
      }}

      function initNetron() {{
        if (typeof netron === 'undefined') {{
          showFallback();
          return;
        }}
        try {{
          var viewer = document.getElementById('viewer');
          if (typeof netron.start === 'function') {{
            netron.start('{onnx_filename}', {{
              container: viewer,
              // Netron options
            }});
            loaded = true;
            var loadingEl = document.getElementById('loading');
            if (loadingEl) loadingEl.style.display = 'none';
          }} else if (typeof netron.view === 'function') {{
            netron.view('{onnx_filename}', {{ container: viewer }});
            loaded = true;
          }} else {{
            showFallback();
          }}
        }} catch (e) {{
          console.error('Netron error:', e);
          showFallback();
        }}
      }}

      // Load Netron from CDN
      var script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/netron@7.6.4/dist/netron.js';
      script.onload = initNetron;
      script.onerror = showFallback;
      document.head.appendChild(script);

      // Timeout fallback (5 seconds)
      setTimeout(showFallback, 5000);
    }})();
  </script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)



# ============================================================
# ✅ Backward-compatibility aliases
# ============================================================

def save_model_architecture(
    model: nn.Module,
    save_dir: str,
    model_name: str = "model",
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
):
    """Backward-compatible wrapper for save_model_with_name."""
    return save_model_with_name(model, save_dir, model_name, input_shape)


def export_model_to_onnx(
    model: nn.Module,
    onnx_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
) -> bool:
    """
    Export just the ONNX (standalone).

    Returns True if successful, False otherwise.
    """
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    try:
        was_training = model.training
        model.eval()

        os.makedirs(os.path.dirname(onnx_path) if os.path.dirname(onnx_path) else ".", exist_ok=True)
        dummy_input = torch.randn(*input_shape, device=model_device)

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
                verbose=False,
            )

        if was_training:
            model.train()

        return os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0

    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")
        return False


# ================================================================
# ✅ JSON UPDATE HELPER
# ================================================================

def update_architecture_info_json(json_path: str, **kwargs) -> None:
    """
    Update the architecture info JSON with additional fields.

    This is used to inject interpretability report paths
    (Captum, LIME, SHAP) into the same JSON that Netron creates.

    Args:
        json_path: Path to the existing architecture_info.json
        **kwargs: Key-value pairs to add/update
    """
    if not os.path.exists(json_path):
        print(f"⚠️ Cannot update JSON (not found): {json_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        # ✅ Add new fields
        info.update(kwargs)
        info["updated_at"] = datetime.now().isoformat()

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

        print(f"✅ Architecture info JSON updated: {json_path}")
        for k in kwargs:
            print(f"   + {k}")

    except Exception as e:
        print(f"⚠️ Failed to update JSON: {e}")

