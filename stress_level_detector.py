import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('stress_ensemble_model.pkl')
scaler = joblib.load('stress_scaler.pkl')

# Features and emojis
features = {
    'anxiety_level': '😰 Anxiety Level (0-20)',
    'depression': '😞 Depression (0-25)',
    'sleep_quality': '😴 Sleep Quality (0-5)',
    'blood_pressure': '🩸 Blood Pressure (1-3)',
    'headache': '🤕 Headache Severity (0-5)',
    'social_support': '🤝 Social Support (0-3)',
    'study_load': '📚 Study Load (0-5)',
    'teacher_student_relationship': '👨‍🏫 Teacher-Student Relation (0-5)',
    'future_career_concerns': '🔮Future Career Concerns (0-5)',
    'peer_pressure': '👥 Peer Pressure (0-5)',
    'extracurricular_activities': '⚽ Extra CurricularActivities Level (0-5)'
}

stress_suggestions = {
    'Low': " Hey buddy !! 👍 You're doing well!\n Keep maintaining a healthy balance and self-care routine.",
    'Moderate': " Heyyyy.....😐\n⚠️ Moderate stress detected.\n Consider relaxation techniques, exercise, and talking to someone you trust.",
    'High': " 🚨 High stress detected! \nIt’s advisable to consult a professional and reduce stressful activities.  "
}

stress_emojis = {
    'Low': "😊",
    'Moderate': "😐",
    'High': "😟"
}

# App window
root = tk.Tk()
root.title("Stress Predictor")
root.geometry("900x900")
root.configure(bg="white")
root.resizable(True, True)

# Heading
title = tk.Label(root, text="💆 Stress Predictor", bg="white", fg="black", font=("Arial", 24, "bold"))
title.pack(pady=10)

# Subtitle
subtitle = tk.Label(root, text="Rate the following factors within range and know your stress level", bg="white", fg="black", font=("Arial", 13))
subtitle.pack(pady=5)

# Scrollable frame
canvas = tk.Canvas(root, bg="white", highlightthickness=0)
form_frame = tk.Frame(canvas, bg="white")
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
canvas.create_window((0, 0), window=form_frame, anchor="n")

def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

form_frame.bind("<Configure>", on_frame_configure)

entries = {}

# Input widgets
style = ttk.Style()
style.configure("RoundedEntry.TEntry", fieldbackground="white", bordercolor="#26c6da", borderwidth=2, relief="flat")

for idx, (key, label_text) in enumerate(features.items()):
    label = tk.Label(form_frame, text=label_text, bg="white", fg="black", font=("Segoe UI Emoji", 11))
    label.grid(row=idx, column=0, pady=(8, 0), padx=20, sticky="ew")

    entry = ttk.Entry(form_frame, width=20, font=("Arial", 11), style="RoundedEntry.TEntry")
    entry.grid(row=idx, column=1, pady=(8, 0), padx=20, ipady=3)
    entries[key] = entry

# Visualize function
def visualize_inputs():
    try:
        user_data = []
        labels = []
        for key in features:
            val = entries[key].get()
            user_data.append(float(val))
            labels.append(features[key].split(' ')[1])

        plt.figure(figsize=(8, 5))
        plt.barh(labels, user_data, color='#26c6da')
        plt.xlabel("Value Rated")
        plt.title("Your Stress Factors Visualization")
        plt.tight_layout()
        plt.show()
    except ValueError:
        result_label.config(text="Please enter valid numbers for all fields.", fg="red")

# Predict function
def predict_stress():
    try:
        user_data = [float(entries[key].get()) for key in features]
        user_np = np.array(user_data).reshape(1, -1)
        user_scaled = scaler.transform(user_np)
        pred = model.predict(user_scaled)[0]
        stress_dict = {0: 'Low', 1: 'Moderate', 2: 'High'}
        stress_level = stress_dict.get(pred, "Unknown")
        suggestion = stress_suggestions.get(stress_level, "")
        emoji = stress_emojis.get(stress_level, "❓")

        result_label.config(
            text=f"{emoji}\n\nYour Stress Level: {stress_level}\n\n{suggestion}",
            fg="black", bg="white", font=("Arial", 20, "bold")
        )
    except Exception as e:
        result_label.config(text="Error in prediction. Please check inputs.", fg="red")

# Hover effect
def on_enter(e):
    e.widget['bg'] = '#00acc1'
def on_leave(e):
    if e.widget['text'].startswith("📊"):
        e.widget['bg'] = '#4dd0e1'
    else:
        e.widget['bg'] = '#0097a7'

# Button Frame
button_frame = tk.Frame(root, bg="white")
button_frame.pack(pady=20)

vis_btn = tk.Button(button_frame, text="Visualize Factors", command=visualize_inputs,
                    bg="#4dd0e1", fg="white", font=("Arial", 11, "bold"),
                    padx=20, pady=8, relief="flat", bd=0)
vis_btn.grid(row=0, column=0, padx=10)
vis_btn.bind("<Enter>", on_enter)
vis_btn.bind("<Leave>", on_leave)

pred_btn = tk.Button(button_frame, text="Predict Stress Level", command=predict_stress,
                     bg="#0097a7", fg="white", font=("Arial", 12, "bold"),
                     padx=20, pady=8, relief="flat", bd=0)
pred_btn.grid(row=0, column=1, padx=10)
pred_btn.bind("<Enter>", on_enter)
pred_btn.bind("<Leave>", on_leave)

# Result Label (BIG and occupying lower half)
result_label = tk.Label(root, text="", bg="white", fg="black", font=("Arial", 20, "bold"),
                        wraplength=800, justify="center")
result_label.pack(pady=30, expand=True, fill="both")

root.mainloop()
