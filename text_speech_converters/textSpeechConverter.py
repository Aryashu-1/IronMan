import tkinter as tk
from gtts import gTTS
import pygame
from io import BytesIO

def text_to_speech():
    text = text_entry.get("1.0", "end-1c")  # Get text from the text entry widget
    if text:
        tts = gTTS(text=text, lang='en-au')  # For Australian English accent

        with BytesIO() as f:
            tts.write_to_fp(f)
            f.seek(0)
            pygame.mixer.init()
            pygame.mixer.music.load(f)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue

# Create the main window
root = tk.Tk()
root.title("Text to Speech")

# Create a text entry widget
text_entry = tk.Text(root, height=10, width=50)
text_entry.pack(pady=10)

# Create a button to trigger text-to-speech conversion
convert_button = tk.Button(root, text="Convert to Speech", command=text_to_speech)
convert_button.pack()

# Run the main event loop
root.mainloop()
