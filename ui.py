import os
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pyaudio
from playsound import playsound
import wave
import threading


def load_model(model_name: str):
    """Load the specified model."""
    if model_name == "emotion2vec":
        from emotion2vecplus import Emotion2Vec

        return Emotion2Vec()
    elif model_name == "cnn":
        from cnn_model import CnnModel

        return CnnModel()
    else:
        raise ValueError(f"Unsupported model: {model_name}")


class EmotionPlotter:
    """Class for recording audio, predicting emotion, and updating plot."""

    def __init__(
        self,
        root,
        update_callback,
        model_name="emotion2vec",
        output_file=r"./Output/input_voice.wav",
    ):
        self.model = load_model(model_name)
        self.root = root
        self.update_callback = update_callback
        self.output_file = output_file
        self.visualizing = False
        self.recording = False
        self.frames = []
        self.setup_gui()
        self.setup_audio()
        self.center_window()

    def setup_gui(self):
        """Setup the graphical user interface."""
        self.root.title("Emotion Prediction")
        frame_color = "#F5F5F5"
        self.root.configure(bg=frame_color)

        self.fig, self.ax = plt.subplots(figsize=(10, 4.5))
        self.configure_plot()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.create_buttons(frame_color)
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)

    def create_buttons(self, frame_color):
        """Create and place buttons in the GUI."""
        button_font = ("Helvetica", 14, "bold")
        button_style = {
            "font": button_font,
            "bg": "#2c3e60",
            "fg": "white",
            "activebackground": "#2c3e55",
            "activeforeground": "white",
            "borderwidth": 0,
            "padx": 10,
            "pady": 5,
        }

        button_frame = tk.Frame(self.root, bg=frame_color)
        button_frame.pack(side=tk.TOP, pady=(10, 10))

        self.record_button = tk.Button(
            button_frame,
            text="Start Recording",
            command=self.start_recording,
            **button_style,
        )
        self.record_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.exit_button = tk.Button(
            button_frame,
            text="Exit",
            command=self.exit_app,
            **button_style,
        )
        self.exit_button.pack(side=tk.RIGHT, padx=10, pady=5)

        self.listen_button = tk.Button(
            button_frame,
            text="Play",
            command=self.play_voice,
            **button_style,
        )
        self.listen_button.pack(side=tk.RIGHT, padx=10, pady=5)
        self.listen_button["state"] = tk.DISABLED

    def setup_audio(self):
        """Configure audio recording settings."""
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.chunk_size = 1024

    def center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def start_recording(self):
        """Start recording audio."""
        self.recording = True
        self.frames = []
        self.record_button.config(text="Stop Recording", command=self.stop_recording)
        self.listen_button["state"] = tk.DISABLED

        # Use a separate thread to handle audio recording
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.start()

    def record_audio(self):
        """Record audio until stopped."""
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        while self.recording:
            data = stream.read(self.chunk_size)
            self.frames.append(data)

        # Stop recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recorded audio
        self.save_audio()

    def stop_recording(self):
        """Stop recording audio and process it."""
        self.recording = False
        self.record_button.config(text="Processing...", state=tk.DISABLED)

        # Wait for the recording thread to finish
        if hasattr(self, "record_thread"):
            self.record_thread.join()
        self.perform_prediction()
        self.record_button.config(
            text="Start Recording", command=self.start_recording, state=tk.NORMAL
        )
        self.listen_button["state"] = tk.NORMAL

    def save_audio(self):
        """Save recorded audio frames to a file."""
        with wave.open(self.output_file, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(self.frames))

    def perform_prediction(self):
        """Perform emotion prediction using the loaded model."""
        try:
            predicted_class, predicted_probs = self.model.predict(self.output_file)
            self.update_callback(self, predicted_class, predicted_probs)
        except Exception as e:
            print(f"An error occurred during prediction: {e}")

    def update_start_text(self, text):
        """Update the text on the start button."""
        self.record_button.config(text=text)
        self.root.update()

    def play_voice(self):
        """Play the recorded audio."""
        playsound(self.output_file)

    def configure_plot(self):
        """Configure the initial plot settings."""
        self.class_names = [
            "Noise",
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        ]
        self.ax.set_title("Emotion Prediction")
        self.ax.set_xlabel("Emotion")
        self.ax.set_ylabel("Probability (%)")
        self.ax.set_ylim(0, 100)
        self.ax.set_xticks(np.arange(len(self.class_names)))
        self.ax.set_xticklabels(self.class_names, rotation=0)

    def update_plot(self, predicted_probs):
        """Update the plot with the predicted probabilities."""
        self.ax.clear()
        bars = self.ax.bar(self.class_names, predicted_probs * 100, color="blue")
        self.configure_plot()
        for i, bar in enumerate(bars):
            self.ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{predicted_probs[i] * 100:.2f}%",
                ha="center",
                va="bottom",
            )
        self.canvas.draw()

    def exit_app(self):
        """Cleanup and exit the application."""
        if self.recording:
            self.stop_recording()  # Ensure recording is stopped
        self.root.quit()  # Exit the Tkinter main loop
        self.root.destroy()  # Destroy the root window


def visualize_emotion_prediction(plotter, predicted_class, predicted_probs):
    """Callback function to update the plot with predictions."""
    plotter.update_plot(predicted_probs)


if __name__ == "__main__":
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.after(0, lambda: root.attributes("-topmost", False))

    plotter = EmotionPlotter(
        root, visualize_emotion_prediction, model_name="emotion2vec"
    )
    root.mainloop()
