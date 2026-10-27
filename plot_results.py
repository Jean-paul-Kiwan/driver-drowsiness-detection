import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# =========================
# LOAD DATA
# =========================
CSV_FILE = "drowsiness_log.csv"
df = pd.read_csv(CSV_FILE)

# =========================
# FIGURE SETUP
# =========================
fig, ax = plt.subplots(figsize=(11, 6))
fig.canvas.manager.set_window_title(
    "Driver Drowsiness Monitoring – Analytics Dashboard"
)

plt.subplots_adjust(left=0.1, bottom=0.35)

# =========================
# INITIAL PLOT
# =========================
current_plot = "EAR"

def plot_ear():
    ax.clear()
    ax.plot(df["time_s"], df["EAR"], label="EAR", color="blue")
    ax.axhline(0.21, linestyle="--", color="red", label="EAR Threshold")
    ax.set_title("Eye Aspect Ratio (EAR) Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EAR")
    ax.legend()
    ax.grid(True)
    fig.canvas.draw_idle()

def plot_mar():
    ax.clear()
    ax.plot(df["time_s"], df["MAR"], label="MAR", color="purple")
    ax.axhline(0.60, linestyle="--", color="red", label="MAR Threshold")
    ax.set_title("Mouth Aspect Ratio (Yawning Detection)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MAR")
    ax.legend()
    ax.grid(True)
    fig.canvas.draw_idle()

def plot_head_pose():
    ax.clear()
    ax.plot(df["time_s"], df["yaw_deg"], label="Yaw", color="green")
    ax.plot(df["time_s"], df["pitch_deg"], label="Pitch", color="orange")
    ax.plot(df["time_s"], df["roll_deg"], label="Roll", color="brown")
    ax.set_title("Head Pose Angles Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Degrees")
    ax.legend()
    ax.grid(True)
    fig.canvas.draw_idle()

def plot_drowsy_state():
    ax.clear()
    ax.plot(df["time_s"], df["drowsy_time_s"], label="Drowsy Time", color="red")
    ax.plot(df["time_s"], df["awake_time_s"], label="Awake Time", color="green")
    ax.set_title("Awake vs Drowsy Accumulated Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Seconds")
    ax.legend()
    ax.grid(True)
    fig.canvas.draw_idle()

def plot_ml_score():
    ax.clear()
    ax.plot(df["time_s"], df["ml_score"], label="ML Score", color="black")
    ax.axhline(0.70, linestyle="--", color="red", label="ML Threshold")
    ax.set_title("Machine Learning Drowsiness Score")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)
    fig.canvas.draw_idle()

# =========================
# BUTTONS
# =========================
ax_ear   = plt.axes([0.05, 0.2, 0.15, 0.07])
ax_mar   = plt.axes([0.22, 0.2, 0.15, 0.07])
ax_head  = plt.axes([0.39, 0.2, 0.15, 0.07])
ax_state = plt.axes([0.56, 0.2, 0.18, 0.07])
ax_ml    = plt.axes([0.77, 0.2, 0.18, 0.07])

btn_ear   = Button(ax_ear,   "EAR")
btn_mar   = Button(ax_mar,   "MAR")
btn_head  = Button(ax_head,  "Head Pose")
btn_state = Button(ax_state, "Awake/Drowsy")
btn_ml    = Button(ax_ml,    "ML Score")

btn_ear.on_clicked(lambda event: plot_ear())
btn_mar.on_clicked(lambda event: plot_mar())
btn_head.on_clicked(lambda event: plot_head_pose())
btn_state.on_clicked(lambda event: plot_drowsy_state())
btn_ml.on_clicked(lambda event: plot_ml_score())

# =========================
# START WITH EAR
# =========================
plot_ear()

plt.show()




#   python plot_results.py 
