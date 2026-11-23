"""Plot measured and analytical centre-of-mass motion during a contact-free force pulse."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd


AUDIT = Path(__file__).resolve().parents[2] / "docs/evidence/chapter3_continuous_planning/p0_a"
COLOURS = {-5: "#2166ac", 0: "#505961", 5: "#bd4d11"}
ROUTES = {"0": "原始右转接近", "baseline": "原始右转接近", "baseline_right": "原始右转接近",
          "1": "向左接近", "signed_left": "向左接近", "2": "向右接近", "signed_right": "向右接近"}
COLUMNS = ["step", "t_s", "com_x_m", "com_y_m", "ideal_x_m", "ideal_y_m",
           "extra_y_m", "ideal_extra_y_m"]


def plot(csv_path, output):
    table = pd.read_csv(csv_path)
    if table.duplicated(["approach", "alpha", "force_n", "step"]).any() or not np.isfinite(table[COLUMNS]).all().all():
        raise ValueError("Each measured force needs distinct steps and finite paired coordinates")
    groups = list(table.groupby(["approach", "alpha"], sort=True))
    if not groups:
        raise ValueError("The trace contains no accepted contact-free intervals")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10, "axes.labelcolor": "#26333c",
                         "axes.edgecolor": "#97a1aa", "xtick.color": "#364450", "ytick.color": "#364450"})
    fig, axes = plt.subplots(len(groups), 2, squeeze=False, figsize=(10.6, 2.03 + 2.45*len(groups)))
    fig.subplots_adjust(left=.105, right=.975, bottom=.105, top=1-1.48/fig.get_figheight(), hspace=.45, wspace=.25)
    fig.text(.105, .975, "横向推力如何改变腾空车辆的质心轨迹", size=17, weight="bold", va="top")
    fig.text(.105, .975-.37/fig.get_figheight(),
             "从施力瞬间算起的 100 毫秒。同一行的三次试验初态相同，逐个 1 毫秒物理步均无接触。", color="#364450", va="top")
    handles = [Line2D([], [], color=colour, lw=2.3, label={-5:"−5 N（世界 −y）", 0:"0 N", 5:"+5 N（世界 +y）"}[force])
               for force,colour in COLOURS.items()]
    handles += [Line2D([], [], color="#29353e", lw=1.6, label="实测"),
                Line2D([], [], color="#29353e", marker="o", markerfacecolor="none", ls="none", ms=6, label="解析值")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.103, .977-.60/fig.get_figheight()),
               frameon=False, ncol=5, columnspacing=1.5, handlelength=1.8, borderaxespad=0)

    bound = 1000 * table[["extra_y_m", "ideal_extra_y_m"]].abs().to_numpy().max()
    for (route_alpha, grp), (xy, delta) in zip(groups, axes, strict=True):
        route,alpha = route_alpha
        if set(grp.force_n) != set(COLOURS):
            raise ValueError(f"The matched −5, 0 and +5 N branches are incomplete for {route_alpha}")
        for force_index,(force, series) in enumerate(grp.groupby("force_n", sort=True)):
            data = series.sort_values("step")
            elapsed = (data.t_s.to_numpy()-data.t_s.iloc[0])*1000
            if not np.allclose(elapsed,np.arange(0,101,5),atol=.003,rtol=0):
                raise ValueError(f"Expected the initial state and twenty 5 ms control steps for {route_alpha}/{force} N")
            colour = COLOURS[force]
            # Stagger the rings so near-coincident curves retain visible analytical samples.
            markers = np.arange(2+force_index, len(data), 4)
            for ax,x,y,analytical_x,analytical_y in [
                    (xy,data.com_x_m,data.com_y_m,data.ideal_x_m,data.ideal_y_m),
                    (delta,elapsed,data.extra_y_m*1000,elapsed,data.ideal_extra_y_m*1000)]:
                ax.plot(x,y,lw=1.7,color=colour)
                ax.plot(analytical_x,analytical_y,ls="none",marker="o",mfc="none",mec=colour,mew=1.05,
                        ms=5.4,markevery=markers,zorder=4)
        key = str(route).removesuffix(".0")
        route_name = ROUTES.get(key, key)
        xy.set_title(f"{route_name}   α = {alpha:g}  |  世界平面质心",loc="left",fontsize=11)
        delta.set_title("相对初始匀速线的横移",loc="left",fontsize=11)
        xy.set(xlabel="世界 x（m）",ylabel="世界 y（m）")
        # Give all three lines the same physical x/y scale without collapsing the axes to a strip.
        xy.set_aspect("equal",adjustable="datalim")
        xy.xaxis.set_major_locator(MaxNLocator(nbins=5))
        xy.yaxis.set_major_locator(MaxNLocator(nbins=4))
        for axis in (xy.xaxis,xy.yaxis):
            axis.set_major_formatter(FormatStrFormatter("%.2f"))
        delta.set(xlim=(-2,102),ylim=(-1.16*bound,1.16*bound),xlabel="施力后时间（ms）",ylabel="额外横移（mm）")
        delta.set_xticks([0,20,40,60,80,100])
        delta.yaxis.set_major_locator(MaxNLocator(nbins=5,steps=[1,2.5,5,10]))
        for ax in (xy,delta):
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(lw=.55,color="#e1e5e9")
            ax.set_axisbelow(True)

    fig.savefig(output,dpi=185,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",type=Path,default=AUDIT/"flight_com_trace.csv")
    parser.add_argument("--output",type=Path,default=AUDIT/"flight_com.png")
    args = parser.parse_args()
    plot(args.csv,args.output)
