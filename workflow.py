# Redesigned workflow diagram with black arrows and spaced layout.

from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

# Setup
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

def draw_box(x, y, w, h, text):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", linewidth=1, facecolor='white', edgecolor='black')
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', wrap=True, fontsize=14)
    return box

def connect(boxA, boxB, locA="bottom", locB="top"):
    coordsA = boxA.get_bbox().get_points()
    coordsB = boxB.get_bbox().get_points()

    if locA == "bottom":
        xyA = ((coordsA[0][0] + coordsA[1][0]) / 2, coordsA[0][1])
    else:
        xyA = ((coordsA[0][0] + coordsA[1][0]) / 2, coordsA[1][1])

    if locB == "top":
        xyB = ((coordsB[0][0] + coordsB[1][0]) / 2, coordsB[1][1])
    else:
        xyB = ((coordsB[0][0] + coordsB[1][0]) / 2, coordsB[0][1])

    ax.add_patch(ConnectionPatch(
        xyA=xyA, xyB=xyB,
        coordsA="data", coordsB="data",
        arrowstyle="->", linewidth=2, color="black"
    ))

# Boxes
b1 = draw_box(2, 12, 6, 1.1, "1) Ground station\n1-min GHI data\n(QC filters applied)")
b2 = draw_box(2, 10, 6, 1.1, "2) Aggregation\n15-min & 60-min averages")
b3 = draw_box(0.7, 8, 3.5, 1.1, "3) Satellite products\n(CAMS, LSA-SAF)\n15-min grids")
b4 = draw_box(5.8, 8, 3.5, 1.1, "4) Reanalysis\n(ERA5, MERRA-2)\nhourly grid means")
b5 = draw_box(2, 6, 6, 1.1, "5) Spatial match\nNearest grid cell & time alignment")
b6 = draw_box(2, 4, 6, 1.1, "6) Filtered dataset\nCompleteness checks\n(day/month thresholds)")
b7 = draw_box(2, 2, 6, 1.1, "7) Error metrics\nrMBE, rMAE, rRMSE\nSeasonal, monthly & SZA analysis")

# Connections (black arrows)
connect(b1, b2)
connect(b2, b5)
connect(b3, b5)
connect(b4, b5)
connect(b5, b6)
connect(b6, b7)

# Title
#ax.set_title("Workflow: QC, Aggregation, Model–Station Matching, and Evaluation", fontsize=14, pad=20)

# Save
path = "./workflow_black_arrows.png"
plt.savefig(path, bbox_inches='tight', dpi=200)
plt.show()
