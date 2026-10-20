"""Plotting helpers shared by Surface and compare()."""
from __future__ import annotations

import uuid

import numpy as np

CLIP_PERCENTILE = 95
LEVELS = 20


def color_range(Zs, zmax=None, log=False):
    """(vmin, vmax) shared by a list of grids, after the optional log transform.

    By default the top of the scale is the 95th percentile of the finite values, so a few
    exploding corners don't wash out the rest of the plot.
    """
    finite = np.concatenate([Z[np.isfinite(Z)].ravel() for Z in Zs])
    vmin = float(finite.min())
    vmax = float(zmax) if zmax is not None else float(np.percentile(finite, CLIP_PERCENTILE))
    if vmax <= vmin:
        vmax = float(finite.max()) if finite.max() > vmin else vmin + 1e-6
    if log:
        floor = _log_floor(finite)
        vmin, vmax = np.log10(max(vmin, floor)), np.log10(max(vmax, floor))
    return vmin, vmax


def _log_floor(values):
    positive = values[values > 0]
    return float(positive.min()) if positive.size else 1e-12


def prepare(Z, vmin, vmax, log=False):
    """Transform and clip Z for display. inf (diverged points) shows as the top of the scale."""
    Z = np.where(np.isfinite(Z), Z, np.inf)
    if log:
        finite = Z[np.isfinite(Z)]
        Z = np.log10(np.maximum(Z, _log_floor(finite)))
    return np.clip(Z, vmin, vmax)


class InteractiveFigure:
    """A plotly figure that displays the same way in Colab, JupyterLab and VS Code.

    Editors render plotly's own output format with their own bundled renderer, which can be
    missing or out of date (blank plots in VS Code). Plain HTML that loads the matching
    plotly.js avoids that. The plotly figure itself is available as `.figure`.
    """

    def __init__(self, figure):
        self.figure = figure

    def _repr_html_(self):
        # Draw only after plotly.js has loaded. Editors like VS Code insert outputs dynamically,
        # where a <script src=...> does not block the script after it, so plotly's own HTML
        # would try to draw before plotly.js exists ("Plotly is not defined").
        from plotly.offline import get_plotlyjs_version

        version = get_plotlyjs_version()
        url = f"https://cdn.plot.ly/plotly-{version}.min.js"
        div = "losslens-" + uuid.uuid4().hex
        fig_json = self.figure.to_json().replace("</", "<\\/")
        height = self.figure.layout.height or 500
        return f"""<div id="{div}" style="width:100%;height:{height}px;"></div>
<script>
(function() {{
  var el = document.getElementById("{div}");
  var fig = {fig_json};
  function draw(P) {{
    try {{ P.newPlot(el, fig.data, fig.layout, {{responsive: true}}); }}
    catch (e) {{ el.innerText = "losslens: could not draw the 3D plot (" + e + ")"; }}
  }}
  if (window.Plotly && window.Plotly.version === "{version}") {{ draw(window.Plotly); return; }}
  if (typeof window.define === "function" && window.define.amd && window.require) {{
    window.require.config({{paths: {{losslens_plotly: "{url[:-3]}"}}}});
    window.require(["losslens_plotly"], draw);
    return;
  }}
  var s = document.createElement("script");
  s.src = "{url}";
  s.onload = function() {{ draw(window.Plotly); }};
  s.onerror = function() {{
    el.innerText = "losslens: could not load plotly.js from cdn.plot.ly. " +
                   "Check the internet connection, or save the plot with .write_html(path).";
  }};
  document.head.appendChild(s);
}})();
</script>"""

    def show(self):
        from IPython.display import display
        display(self)

    def write_html(self, path):
        """Save as a standalone web page you can open in any browser, even offline."""
        self.figure.write_html(path, include_plotlyjs=True)


def loss_label(log):
    return "log10(loss)" if log else "loss"


def draw_contour(ax, surf, vmin, vmax, log=False, title=None):
    Z = prepare(surf.Z, vmin, vmax, log)
    levels = np.linspace(vmin, vmax, LEVELS + 1)
    filled = ax.contourf(surf.x, surf.y, Z, levels=levels, cmap="viridis", extend="neither")
    lines = ax.contour(surf.x, surf.y, Z, levels=levels[::4], colors="white",
                       linewidths=0.6, alpha=0.7)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.3g")
    ax.plot(0, 0, marker="*", color="red", markersize=12, markeredgecolor="white",
            label="trained model")
    ax.set_xlabel("direction 1")
    ax.set_ylabel("direction 2")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return filled
