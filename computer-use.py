"""
Computer Vision-Based Object Location Tool
=======================================

This script uses Moondream's REST API to locate objects on your screen and visualize their position.
It creates a visual overlay to show where the detected object is located.

Usage:
------
    1. Run the script: python computer-use.py
    2. Press Alt+M for text search or Alt+N for voice search
    3. Enter what you want to find
    4. Wait for the countdown
    5. The object location will be highlighted on screen

Features:
--------
    - Text and voice search interfaces
    - Hotkey-triggered activation
    - Visual overlay with crosshair and coordinates
    - White outline for visibility on any background
    - Coordinate display in pixels

Requirements:
-----------
    - requests: For API communication
    - PIL (Pillow): For image handling
    - pyautogui: For screen capture and dimensions
    - pynput: For global hotkeys
    - tkinter: For visualization overlay
    - vosk: For voice recognition

To install the dependencies, run:
    pip install requests Pillow pyautogui pynput vosk
"""

import os
import requests
from PIL import Image
import pyautogui
import time
import io
import sys
import tkinter as tk
from tkinter import ttk
import base64
from io import BytesIO
import vosk
import json
import pyaudio
import queue
import threading
from typing import Optional, List, Tuple, Dict
from pynput import keyboard

# Configuration constants
CONFIG = {
    'WINDOW': {
        'WIDTH': 800,
        'HEIGHT': 80,
        'VOICE_HEIGHT': 120,
    },
    'COLORS': {
        'BG': '#1C1C1C',
        'TEXT': '#FFFFFF',
        'PLACEHOLDER': '#6E7681',
        'BORDER': '#333333',
        'HIGHLIGHT': '#2F81F7',
        'SHADOW': '#000000',
    },
    'FONTS': {
        'MAIN': ('Segoe UI', 16),
        'ICON': ('Segoe UI', 14),
        'HINT': ('Segoe UI', 9),
        'SECONDARY': ('Segoe UI', 12),
    },
    'API': {
        'KEY': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI0YWQ5ODNkZC02NWIyLTRkNzMtYmMwNy1lMGQ0YWVkNWFjMzIiLCJpYXQiOjE3MzQwNTAwMzh9.GKoEaYT2_AjB6e9ZL3pczGygnWSjl7GKC08ZCJkaIVM',
        'URL': 'https://api.moondream.ai/v1/point',
    },
    'VOICE': {
        'SAMPLE_RATE': 16000,
        'CHUNK_SIZE': 8000,
        'READ_SIZE': 4000,
    },
}

class BaseSearchWindow:
    """Base class for search windows with common UI elements."""
    
    def __init__(self, window_height: int = CONFIG['WINDOW']['HEIGHT']):
        self.root = tk.Tk()
        self.root.title("")
        self.setup_window(window_height)
        self.setup_container()
        self.result = None
        
    def setup_window(self, window_height: int) -> None:
        """Configure window properties."""
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)
        self.root.lift()
        
        # Center window
        window_width = CONFIG['WINDOW']['WIDTH']
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/3)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.root.configure(bg=CONFIG['COLORS']['BG'])
        
    def setup_container(self) -> None:
        """Set up the main container frame."""
        self.container = tk.Frame(
            self.root,
            bg=CONFIG['COLORS']['BG'],
            highlightbackground=CONFIG['COLORS']['BORDER'],
            highlightthickness=1,
        )
        self.add_shadow()
        self.container.pack(fill='both', expand=True, padx=1, pady=1)
        
    def add_shadow(self) -> None:
        """Add drop shadow effect to window."""
        shadow_size = 5
        for i in range(shadow_size):
            alpha = 0.1 - (i * 0.02)
            shadow = tk.Frame(
                self.root,
                bg=CONFIG['COLORS']['SHADOW'],
                height=2,
            )
            shadow.place(x=0, y=shadow_size+i, relwidth=1)
            shadow.lift()
            
    def add_shortcut_hint(self, text: str) -> None:
        """Add shortcut hint label."""
        self.shortcut_label = tk.Label(
            self.container,
            text=text,
            font=CONFIG['FONTS']['HINT'],
            fg=CONFIG['COLORS']['PLACEHOLDER'],
            bg=CONFIG['COLORS']['BG']
        )
        self.shortcut_label.pack(side='right', padx=(0, 15), pady=0)
        
    def get_search_term(self) -> Optional[str]:
        """Run the window and return the search term."""
        self.root.mainloop()
        return self.result

def countdown(seconds):
    """Display a countdown timer in the console before taking screenshot.
    
    Args:
        seconds (int): Number of seconds to count down
    
    The countdown is displayed inline in the console with a carriage return,
    providing a clean visual feedback to the user.
    """
    for i in range(seconds, 0, -1):
        sys.stdout.write(f'\rTaking screenshot in {i} seconds...')
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write('\rTaking screenshot now!      \n')
    sys.stdout.flush()

class ScreenHandler:
    """Class for handling screen capture and coordinate operations."""
    
    @staticmethod
    def take_screenshot() -> bytes:
        """Capture the current screen.
        
        Returns:
            bytes: Screenshot image data in PNG format
        """
        screenshot = pyautogui.screenshot()
        img_byte_arr = BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    
    @staticmethod
    def get_coordinates(image_bytes: bytes, target_object: str) -> List[Tuple[int, int]]:
        """Get coordinates using Moondream API.
        
        Args:
            image_bytes: Screenshot image data
            target_object: Object to locate in the image
            
        Returns:
            List of (x, y) coordinate tuples
            
        Raises:
            ValueError: If API key is missing or no coordinates found
            requests.exceptions.RequestException: If API request fails
        """
        api_key = CONFIG['API']['KEY']
        if not api_key:
            raise ValueError("Please set MOONDREAM_API_KEY environment variable")
        
        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        headers = {
            "X-Moondream-Auth": api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "image_url": f"data:image/jpeg;base64,{base64_image}",
            "object": target_object,
            "stream": False
        }
        
        try:
            response = requests.post(CONFIG['API']['URL'], headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            print(f"API Response: {result}")
            
            if not result or 'points' not in result or not result['points']:
                raise ValueError(f"No coordinates found for '{target_object}'")
            
            # Convert normalized coordinates to screen coordinates
            screen_width, screen_height = pyautogui.size()
            print(f"Screen size: {screen_width}x{screen_height}")
            
            screen_points = []
            for point in result['points']:
                x = int(float(point["x"]) * screen_width)
                y = int(float(point["y"]) * screen_height)
                screen_points.append((x, y))
            
            print(f"Found {len(screen_points)} possible locations")
            return screen_points
            
        except requests.exceptions.RequestException as e:
            print(f"API Request Failed: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response content: {e.response.text}")
            raise
    
    @staticmethod
    def touch_click(x: int, y: int, radius: int = 20) -> None:
        """Perform a touch-like click by clicking corners of a square around target.
        
        Args:
            x: X coordinate to click
            y: Y coordinate to click
            radius: Radius of the click area
        """
        click_points = [
            (x, y),  # Center
            (x - radius, y - radius),  # Top left
            (x + radius, y - radius),  # Top right
            (x - radius, y + radius),  # Bottom left
            (x + radius, y + radius),  # Bottom right
        ]
        
        # Move to first position
        pyautogui.moveTo(click_points[0][0], click_points[0][1])
        
        # Click all points in rapid succession
        for px, py in click_points:
            pyautogui.click(px, py, _pause=False)
    
    @staticmethod
    def type_text(text: str) -> None:
        """Type text at current cursor position with a natural delay.
        
        Args:
            text: Text to type
        """
        time.sleep(0.3)  # Let UI settle
        pyautogui.write(text, interval=0.02)

class CoordinateVisualizer:
    """Class for visualizing coordinates on screen with interactive selection."""
    
    def __init__(self, points: List[Tuple[int, int]]):
        self.points = points
        self.marker_size = 30
        self.pulse_items = []
        self.selected_coords = None
        self.root = tk.Tk()
        self.setup_window()
        
    def setup_window(self) -> None:
        """Set up the visualization window."""
        self.root.attributes('-alpha', 0.7)
        self.root.attributes('-topmost', True)
        self.root.attributes('-fullscreen', True)
        self.root.lift()
        
        # Make window click-through
        self.root.attributes('-transparentcolor', 'black')
        self.root.config(bg='black')
        self.root.wm_attributes('-disabled', True)
        
        # Create canvas
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg='black')
        self.canvas.pack(fill='both', expand=True)
        
        # Create coordinate mapping
        self.coord_map = {}
        for idx, (x, y) in enumerate(self.points):
            letter = chr(65 + idx)
            self.coord_map[letter.lower()] = (x, y)
            self.coord_map[letter.upper()] = (x, y)
            self.draw_marker(x, y, letter)
        
        # Add instructions
        self.add_instructions()
        
        # Bind events
        self.root.bind('<Key>', self.on_key)
        self.root.focus_force()
        
        # Start animation
        self.pulse_size = 0
        self.pulse_growing = True
        self.animate_pulse()
    
    def draw_marker(self, x: int, y: int, letter: str) -> None:
        """Draw a single coordinate marker."""
        # Outer white glow for visibility
        self.canvas.create_oval(
            x-self.marker_size-4, y-self.marker_size-4,
            x+self.marker_size+4, y+self.marker_size+4,
            outline='white', width=6
        )
        self.canvas.create_line(
            x-self.marker_size*2-4, y,
            x+self.marker_size*2+4, y,
            fill='white', width=6
        )
        self.canvas.create_line(
            x, y-self.marker_size*2-4,
            x, y+self.marker_size*2+4,
            fill='white', width=6
        )
        
        # Red crosshair
        self.canvas.create_oval(
            x-self.marker_size, y-self.marker_size,
            x+self.marker_size, y+self.marker_size,
            outline='red', width=4
        )
        self.canvas.create_line(
            x-self.marker_size*2, y,
            x+self.marker_size*2, y,
            fill='red', width=4
        )
        self.canvas.create_line(
            x, y-self.marker_size*2,
            x, y+self.marker_size*2,
            fill='red', width=4
        )
        
        # Pulsing circle
        pulse = self.canvas.create_oval(
            x-self.marker_size-2, y-self.marker_size-2,
            x+self.marker_size+2, y+self.marker_size+2,
            outline='#FF4444', width=3
        )
        self.pulse_items.append((pulse, x, y))
        
        # Letter and coordinates
        self.canvas.create_text(
            x, y - self.marker_size*2 - 25,
            text=letter,
            fill='yellow',
            font=('Arial', 28, 'bold')
        )
        self.canvas.create_text(
            x, y + self.marker_size*2 + 25,
            text=f"({x}, {y})",
            fill='white',
            font=('Arial', 14)
        )
    
    def add_instructions(self) -> None:
        """Add instruction text to the visualization."""
        instructions = "Press letter (A-{}) to select point to click, ESC to cancel".format(
            chr(64 + len(self.points))
        )
        self.canvas.create_text(
            self.root.winfo_screenwidth()//2, 50,
            text=instructions,
            fill='white',
            font=('Arial', 18)
        )
    
    def animate_pulse(self) -> None:
        """Animate the pulsing circles."""
        if self.pulse_growing:
            self.pulse_size += 1
            if self.pulse_size >= 20:
                self.pulse_growing = False
        else:
            self.pulse_size -= 1
            if self.pulse_size <= 0:
                self.pulse_growing = True
        
        for pulse, x, y in self.pulse_items:
            self.canvas.coords(
                pulse,
                x-self.marker_size-2-self.pulse_size,
                y-self.marker_size-2-self.pulse_size,
                x+self.marker_size+2+self.pulse_size,
                y+self.marker_size+2+self.pulse_size
            )
        
        if self.root.winfo_exists():
            self.root.after(20, self.animate_pulse)
    
    def on_key(self, event: tk.Event) -> None:
        """Handle key press events."""
        if event.char in self.coord_map:
            self.selected_coords = self.coord_map[event.char]
            self.root.quit()
        elif event.keysym == 'Escape':
            self.root.quit()
    
    def show(self) -> Optional[Tuple[int, int]]:
        """Show the visualization and return selected coordinates."""
        self.root.mainloop()
        self.root.destroy()
        
        if self.selected_coords:
            time.sleep(0.05)
            ScreenHandler.touch_click(*self.selected_coords)
        
        return self.selected_coords

class ActionHandler:
    """Class for handling different types of user actions."""
    
    @staticmethod
    def handle_action(search_term: str) -> bool:
        """Handle different types of actions based on search term."""
        if ' and type ' in search_term.lower():
            return ActionHandler._handle_find_and_type(search_term)
        elif search_term.lower().startswith('type '):
            return ActionHandler._handle_type(search_term[5:].strip())
        else:
            return ActionHandler._handle_find(search_term)
    
    @staticmethod
    def _handle_find_and_type(search_term: str) -> bool:
        """Handle combined find and type action."""
        parts = search_term.lower().split(' and type ')
        find_part = parts[0]
        type_part = parts[1].strip()
        
        # Extract target object
        target = find_part[5:].strip() if find_part.startswith('find ') else find_part.strip()
        
        # First find and click the target
        screenshot_bytes = ScreenHandler.take_screenshot()
        print(f"Looking for: {target}")
        
        try:
            points = ScreenHandler.get_coordinates(screenshot_bytes, target)
            print(f"Found {len(points)} possible locations for {target}")
            
            visualizer = CoordinateVisualizer(points)
            if visualizer.show():
                ScreenHandler.type_text(type_part)
                return True
        except Exception as e:
            print(f"Error during find and type: {str(e)}")
        
        return False
    
    @staticmethod
    def _handle_type(text: str) -> bool:
        """Handle type action."""
        screenshot_bytes = ScreenHandler.take_screenshot()
        print("Looking for text box...")
        
        try:
            points = ScreenHandler.get_coordinates(screenshot_bytes, "text box")
            print(f"Found {len(points)} possible text box locations")
            
            visualizer = CoordinateVisualizer(points)
            if visualizer.show():
                ScreenHandler.type_text(text)
                return True
        except Exception as e:
            print(f"Error during type: {str(e)}")
        
        return False
    
    @staticmethod
    def _handle_find(search_term: str) -> bool:
        """Handle find action."""
        # Extract target from search term
        target = search_term[5:].strip() if search_term.lower().startswith('find ') else search_term
        
        screenshot_bytes = ScreenHandler.take_screenshot()
        print(f"Looking for: {target}")
        
        try:
            points = ScreenHandler.get_coordinates(screenshot_bytes, target)
            print(f"Found {len(points)} possible locations for {target}")
            
            visualizer = CoordinateVisualizer(points)
            return visualizer.show() is not None
        except Exception as e:
            print(f"Error during find: {str(e)}")
        
        return False

class SearchWindow(BaseSearchWindow):
    """Modern Vercel-style command palette."""
    
    def __init__(self):
        super().__init__(CONFIG['WINDOW']['HEIGHT'])
        
        # Search icon (⌘)
        self.icon_label = tk.Label(
            self.container,
            text="⌘",
            font=CONFIG['FONTS']['ICON'],
            fg=CONFIG['COLORS']['PLACEHOLDER'],
            bg=CONFIG['COLORS']['BG']
        )
        self.icon_label.pack(side='left', padx=(15, 5), pady=0)
        
        # Custom Entry widget
        self.search_var = tk.StringVar()
        self.entry = tk.Entry(
            self.container,
            textvariable=self.search_var,
            font=CONFIG['FONTS']['MAIN'],
            fg=CONFIG['COLORS']['TEXT'],
            bg=CONFIG['COLORS']['BG'],
            insertbackground=CONFIG['COLORS']['TEXT'],
            relief='flat',
            highlightthickness=0,
            bd=0
        )
        self.entry.pack(fill='x', expand=True, padx=(5, 15), pady=(20, 20))
        
        # Add placeholder
        self.entry.insert(0, "Type to find anything...")
        self.entry.config(fg=CONFIG['COLORS']['PLACEHOLDER'])
        
        # Bind events
        self.entry.bind('<FocusIn>', self.on_entry_click)
        self.entry.bind('<FocusOut>', self.on_focus_out)
        self.root.bind('<Return>', lambda e: self.submit())
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        
        # Add shortcut hint
        self.add_shortcut_hint("alt+m to open, esc to close")
        
        # Ensure focus
        self.root.update_idletasks()
        self.root.lift()
        self.entry.focus_force()
        
        # Keep focused
        self._focus_check_id = None
        self.ensure_focus()
    
    def ensure_focus(self):
        """Ensure window stays focused."""
        if self.root.winfo_exists():
            self.root.lift()
            self.entry.focus_force()
            self._focus_check_id = self.root.after(100, self.ensure_focus)
    
    def on_entry_click(self, event):
        """Handle entry field click."""
        if self.entry.get() == "Type to find anything...":
            self.entry.delete(0, tk.END)
            self.entry.config(fg=CONFIG['COLORS']['TEXT'])
    
    def on_focus_out(self, event):
        """Handle focus out."""
        if not self.entry.get():
            self.entry.insert(0, "Type to find anything...")
            self.entry.config(fg=CONFIG['COLORS']['PLACEHOLDER'])
    
    def submit(self):
        """Handle submission."""
        text = self.search_var.get()
        if text and text != "Type to find anything...":
            self.result = text
            if self._focus_check_id:
                self.root.after_cancel(self._focus_check_id)
            self.root.destroy()
    
    def get_search_term(self):
        """Run the window and return the search term."""
        try:
            self.root.mainloop()
        finally:
            if self._focus_check_id:
                self.root.after_cancel(self._focus_check_id)
        return self.result

class VoiceRecognizer:
    """Class for handling voice recognition functionality."""
    
    def __init__(self, callback):
        self.callback = callback
        self.running = True
        self.recognition_thread = None
        self.last_text = ""
        self.last_speech_time = time.time()
        self.silence_threshold = 0.8  # Reduced from 1.5 to 0.8 seconds for faster response
    
    def start(self):
        """Start voice recognition in a separate thread."""
        self.running = True
        self.recognition_thread = threading.Thread(target=self._recognize_speech)
        self.recognition_thread.daemon = True  # Make thread daemon so it exits with main program
        self.recognition_thread.start()
    
    def stop(self):
        """Stop voice recognition."""
        self.running = False
        # Don't join the thread if we're in it
        if (self.recognition_thread and 
            threading.current_thread() is not self.recognition_thread):
            self.recognition_thread.join(timeout=1.0)  # Wait up to 1 second
    
    def _recognize_speech(self):
        """Run voice recognition in background thread."""
        try:
            model = vosk.Model(lang="en-us")
            
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=CONFIG['VOICE']['SAMPLE_RATE'],
                input=True,
                frames_per_buffer=CONFIG['VOICE']['CHUNK_SIZE']
            )
            
            recognizer = vosk.KaldiRecognizer(model, CONFIG['VOICE']['SAMPLE_RATE'])
            
            while self.running:
                try:
                    data = stream.read(CONFIG['VOICE']['READ_SIZE'], exception_on_overflow=False)
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip()
                        
                        # If we got text, update the last speech time
                        if text:
                            self.last_speech_time = time.time()
                            self.last_text = text
                            self.callback(text, final=False)
                        # If we've had silence for a while and have previous text
                        elif self.last_text and (time.time() - self.last_speech_time) > self.silence_threshold:
                            self.callback(self.last_text, final=True)
                            self.last_text = ""
                            
                except Exception as e:
                    print(f"Error reading audio data: {str(e)}")
                    break
        
        except Exception as e:
            print(f"Error during voice recognition: {str(e)}")
        finally:
            self.running = False
            if 'stream' in locals():
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
            if 'audio' in locals():
                try:
                    audio.terminate()
                except:
                    pass

class VoiceSearchWindow(BaseSearchWindow):
    """Voice-controlled command palette for accessibility."""
    
    def __init__(self):
        super().__init__(CONFIG['WINDOW']['VOICE_HEIGHT'])
        
        # Microphone icon
        self.icon_label = tk.Label(
            self.container,
            text="🎤",
            font=CONFIG['FONTS']['ICON'],
            fg=CONFIG['COLORS']['TEXT'],
            bg=CONFIG['COLORS']['BG']
        )
        self.icon_label.pack(side='left', padx=(15, 5), pady=0)
        
        # Status label
        self.status_label = tk.Label(
            self.container,
            text="Listening...",
            font=CONFIG['FONTS']['MAIN'],
            fg=CONFIG['COLORS']['TEXT'],
            bg=CONFIG['COLORS']['BG']
        )
        self.status_label.pack(fill='x', expand=True, padx=(5, 15), pady=(10, 0))
        
        # Recognized text label
        self.text_label = tk.Label(
            self.container,
            text="",
            font=CONFIG['FONTS']['SECONDARY'],
            fg=CONFIG['COLORS']['PLACEHOLDER'],
            bg=CONFIG['COLORS']['BG'],
            wraplength=CONFIG['WINDOW']['WIDTH']-60
        )
        self.text_label.pack(fill='x', expand=True, padx=(5, 15), pady=(0, 10))
        
        # Add shortcut hint
        self.add_shortcut_hint("esc to cancel")
        
        # Add loading spinner frames
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = 0
        
        # Bind escape and window close
        self.root.bind('<Escape>', lambda e: self.stop_recognition())
        self.root.protocol("WM_DELETE_WINDOW", self.stop_recognition)
        
        # Initialize voice recognition
        self.recognizer = VoiceRecognizer(self._on_voice_result)
        self.recognizer.start()
        
        # Keep focused
        self._focus_check_id = None
        self.ensure_focus()
    
    def ensure_focus(self):
        """Ensure window stays focused."""
        if self.root.winfo_exists():
            self.root.lift()
            self._focus_check_id = self.root.after(100, self.ensure_focus)
    
    def _on_voice_result(self, text, final=False):
        """Handle voice recognition results."""
        if not self.root.winfo_exists():
            return
            
        # Always update the display text first
        self.text_label.config(text=text)
        self.root.update()  # Force update the display
        
        # Only process the command if it's final or contains a trigger word
        if final and ("find" in text.lower() or "type" in text.lower()):
            # Show loading animation
            self.animate_loading(text)
            self.result = text
            self.stop_recognition()
    
    def animate_loading(self, text, frames=6):
        """Animate a loading spinner next to text."""
        for _ in range(frames):
            if not self.root.winfo_exists():
                break
            spinner = self.spinner_frames[self.spinner_idx]
            self.text_label.config(text=f"{text} {spinner}")
            self.root.update()
            self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)
            time.sleep(0.05)
    
    def stop_recognition(self):
        """Stop voice recognition and close window."""
        if self._focus_check_id:
            self.root.after_cancel(self._focus_check_id)
        self.recognizer.stop()
        if self.root.winfo_exists():
            self.root.destroy()
    
    def get_search_term(self):
        """Run the window and return the recognized command."""
        try:
            self.root.mainloop()
        finally:
            if self._focus_check_id:
                self.root.after_cancel(self._focus_check_id)
            self.recognizer.stop()
        return self.result

def main():
    """Main function handling the program flow."""
    print("\nComputer Vision-Based Object Location Tool")
    print("----------------------------------------")
    print("Voice Recognition: Using default model")
    
    print("\nControls:")
    print("- Press 'Alt+M' for text search")
    print("- Press 'Alt+N' for voice control")
    print("- Press Ctrl+C to exit\n")
    
    # Create an event to signal when to exit
    exit_event = threading.Event()
    
    # Track key states with a dictionary to handle complex key combinations
    key_states = {
        'alt_l': False,        # State of left Alt key
        'alt_r': False,        # State of right Alt key
        'last_activation': 0,  # Timestamp of last hotkey activation
        'tab_pressed': False,  # Whether Tab was pressed during this Alt press
        'other_key_pressed': False,  # Whether any other key was pressed during this Alt press
    }
    
    def on_press(key):
        """Handle key press events.
        
        This function implements the following logic:
        1. Prevents rapid-fire activation by checking time since last activation
        2. Tracks Alt key state (both left and right Alt)
        3. Resets state when Alt is pressed to allow new combinations
        4. Tracks if Tab or other keys are pressed while Alt is down
        5. Only triggers on clean Alt+M/N combinations (no other keys pressed)
        """
        try:
            current_time = time.time()
            # Prevent rapid-fire activation (must wait 0.5s between activations)
            if current_time - key_states['last_activation'] < 0.5:
                return
                
            # When Alt is pressed, reset the state to allow new combinations
            if key in (keyboard.Key.alt_l, keyboard.Key.alt):  # Note: some systems report alt_l as just alt
                key_states['alt_l'] = True
                key_states['other_key_pressed'] = False  # Reset other key flag
                key_states['tab_pressed'] = False  # Reset tab flag
            elif key == keyboard.Key.alt_r:
                key_states['alt_r'] = True
                key_states['other_key_pressed'] = False  # Reset other key flag
                key_states['tab_pressed'] = False  # Reset tab flag
            # If Tab is pressed while Alt is down, mark both Tab and other key as pressed
            elif key == keyboard.Key.tab:
                key_states['tab_pressed'] = True
                key_states['other_key_pressed'] = True  # Tab counts as another key
            # Only check for M/N if Alt is down and no other keys were pressed
            elif (key_states['alt_l'] or key_states['alt_r']) and not key_states['other_key_pressed']:
                if hasattr(key, 'char') and key.char:  # Ensure it's a character key
                    if key.char.lower() == 'm':
                        key_states['last_activation'] = current_time
                        key_states['other_key_pressed'] = True  # Prevent further activations
                        on_activate_text()
                    elif key.char.lower() == 'n':
                        key_states['last_activation'] = current_time
                        key_states['other_key_pressed'] = True  # Prevent further activations
                        on_activate_voice()
                    else:
                        # Any other character key pressed while Alt is down
                        key_states['other_key_pressed'] = True
        except AttributeError:
            # Any non-character key pressed while Alt is down
            key_states['other_key_pressed'] = True
    
    def on_release(key):
        """Handle key release events.
        
        This function:
        1. Resets Alt key states when released
        2. Resets the "other key pressed" flag when Alt is released
        3. Resets Tab state when Tab is released
        """
        if key in (keyboard.Key.alt_l, keyboard.Key.alt):
            key_states['alt_l'] = False
            key_states['other_key_pressed'] = False  # Reset when Alt is released
        elif key == keyboard.Key.alt_r:
            key_states['alt_r'] = False
            key_states['other_key_pressed'] = False  # Reset when Alt is released
        elif key == keyboard.Key.tab:
            key_states['tab_pressed'] = False
    
    def on_activate_text():
        """Handle text search activation."""
        try:
            search_window = SearchWindow()
            search_term = search_window.get_search_term()
            if search_term:
                try:
                    ActionHandler.handle_action(search_term)
                except Exception as e:
                    print(f"Error handling action: {str(e)}")
        except Exception as e:
            print(f"Error in text search: {str(e)}")
    
    def on_activate_voice():
        """Handle voice search activation."""
        try:
            voice_window = VoiceSearchWindow()
            search_term = voice_window.get_search_term()
            if search_term:
                try:
                    ActionHandler.handle_action(search_term)
                except Exception as e:
                    print(f"Error handling action: {str(e)}")
        except Exception as e:
            print(f"Error in voice search: {str(e)}")
    
    # Set up keyboard listeners
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    listener.start()
    
    try:
        # Keep the main thread alive until Ctrl+C
        exit_event.wait()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
    finally:
        listener.stop()

if __name__ == "__main__":
    main()
