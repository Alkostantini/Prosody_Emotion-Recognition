import pyaudio
import wave
import tkinter as tk
import threading
import os

class AudioRecorder:
    def __init__(self, duration=2.5, sample_rate=44100, output_folder="recordings"):
        self.duration = duration
        self.sample_rate = sample_rate
        self.output_folder = output_folder
        self.recordings_count = 0

    def record_audio(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=1024)
        print("Recording...")
        frames = []
        for _ in range(0, int(self.sample_rate / 1024 * self.duration)):
            data = stream.read(1024)
            frames.append(data)
        print("Finished recording.")
        stream.stop_stream()
        stream.close()
        audio.terminate()

        filename = os.path.join(self.output_folder, f"audio{self.recordings_count + 1}.wav")
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.recordings_count += 1

class App:
    def __init__(self, master):
        self.master = master
        self.recorder = AudioRecorder()

        self.master.geometry("300x100")
        self.start_button = tk.Button(master, text="Start Recording", command=self.start_recording)
        self.start_button.pack()

    def start_recording(self):
        self.start_button.config(state=tk.DISABLED)
        self.recording_label = tk.Label(self.master, text="Recording...", fg="red")
        self.recording_label.pack()
        threading.Thread(target=self.record).start()

    def record(self):
        self.recorder.record_audio()
        self.recording_label.config(text="Finished recording.", fg="green")
        self.start_button.config(state=tk.NORMAL)

root = tk.Tk()
app = App(root)
root.mainloop()
