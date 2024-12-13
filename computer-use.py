"""
Computer Vision-Based Object Location Tool
=======================================

This script uses Moondream's REST API to locate objects on your screen and visualize their position.
It creates a visual overlay to show where the detected object is located.

Usage:
------
    1. Run the script: python computer-use.py
    2. Press ` (backtick) to open the search window
    3. Enter what you want to find
    4. Wait for the countdown
    5. The object location will be highlighted on screen

Features:
--------
    - Floating search window
    - Hotkey-triggered activation
    - Countdown timer before capture
    - Visual overlay with crosshair and coordinates
    - White outline for visibility on any background
    - Coordinate display in pixels

Requirements:
-----------
    - requests: For API communication
    - PIL (Pillow): For image handling
    - pyautogui: For screen capture and dimensions
    - keyboard: For hotkey detection
    - tkinter: For visualization overlay

To install the dependencies, run:
    pip install requests Pillow pyautogui keyboard

Note: Requires a valid Moondream API key
"""

import os
import requests
from PIL import Image
import pyautogui
import keyboard
import time
import io
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import Canvas
import base64
from io import BytesIO
import vosk
import json
import pyaudio
import queue
import threading

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

def touch_click(x, y, radius=20):
    """Perform a touch-like click by clicking corners of a square around target simultaneously."""
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

def visualize_coordinates(points, duration=None):
    """Create a transparent overlay window showing multiple target coordinates."""
    root = tk.Tk()
    root.attributes('-alpha', 0.7)
    root.attributes('-topmost', True)
    root.attributes('-fullscreen', True)
    root.lift()
    
    # Make window click-through
    root.attributes('-transparentcolor', 'black')
    root.config(bg='black')
    root.wm_attributes('-disabled', True)
    
    # Create canvas with transparent background
    canvas = Canvas(root, highlightthickness=0, bg='black')
    canvas.pack(fill='both', expand=True)
    
    # Force focus
    def ensure_focus():
        root.focus_force()
        root.lift()
        if root.winfo_exists():  # Check if window still exists
            root.after(100, ensure_focus)
    
    # Start focus checking after a brief delay
    root.after(10, ensure_focus)
    
    marker_size = 30
    selected_coords = None
    pulse_items = []
    
    # Create a mapping of letters to coordinates
    coord_map = {}
    for idx, (x, y) in enumerate(points):
        letter = chr(65 + idx)
        coord_map[letter.lower()] = (x, y)
        coord_map[letter.upper()] = (x, y)
        
        # Draw outer glow (white outline for visibility)
        outer_glow = canvas.create_oval(x-marker_size-4, y-marker_size-4, 
                                      x+marker_size+4, y+marker_size+4, 
                                      outline='white', width=6)  # Thicker outline
        canvas.create_line(x-marker_size*2-4, y, x+marker_size*2+4, y, 
                         fill='white', width=6)  # Thicker lines
        canvas.create_line(x, y-marker_size*2-4, x, y+marker_size*2+4, 
                         fill='white', width=6)
        
        # Draw main crosshair (red)
        inner_circle = canvas.create_oval(x-marker_size, y-marker_size, 
                                        x+marker_size, y+marker_size, 
                                        outline='red', width=4)  # Thicker outline
        canvas.create_line(x-marker_size*2, y, x+marker_size*2, y, 
                         fill='red', width=4)
        canvas.create_line(x, y-marker_size*2, x, y+marker_size*2, 
                         fill='red', width=4)
        
        # Add pulsing circle
        pulse = canvas.create_oval(x-marker_size-2, y-marker_size-2,
                                 x+marker_size+2, y+marker_size+2,
                                 outline='#FF4444', width=3)
        pulse_items.append((pulse, x, y))

        # Add letter and coordinate text
        canvas.create_text(x, y - marker_size*2 - 25,  # Adjusted for larger marker
                         text=f"{letter}", 
                         fill='yellow', 
                         font=('Arial', 28, 'bold'))  # Larger font
        canvas.create_text(x, y + marker_size*2 + 25, 
                         text=f"({x}, {y})", 
                         fill='white', 
                         font=('Arial', 14))  # Larger font

    # Pulse animation
    pulse_size = 0
    pulse_growing = True
    def animate_pulse():
        nonlocal pulse_size, pulse_growing
        
        if pulse_growing:
            pulse_size += 1
            if pulse_size >= 20:  # Max pulse size
                pulse_growing = False
        else:
            pulse_size -= 1
            if pulse_size <= 0:
                pulse_growing = True
        
        # Update all pulse circles
        for pulse, x, y in pulse_items:
            canvas.coords(pulse,
                        x-marker_size-2-pulse_size, y-marker_size-2-pulse_size,
                        x+marker_size+2+pulse_size, y+marker_size+2+pulse_size)
        
        root.after(20, animate_pulse)  # Update every 20ms

    # Start pulse animation
    animate_pulse()
    
    def on_key(event):
        nonlocal selected_coords
        key = event.char
        if key in coord_map:
            selected_coords = coord_map[key]
            root.quit()
        elif event.keysym == 'Escape':
            root.quit()

    # Add instructions
    instructions = "Press letter (A-{}) to select point to click, ESC to cancel".format(
        chr(64 + len(points))
    )
    canvas.create_text(root.winfo_screenwidth()//2, 50,
                      text=instructions,
                      fill='white',
                      font=('Arial', 18))  # Larger font

    # Bind keyboard events
    root.bind('<Key>', on_key)
    root.focus_force()

    # Run the window
    root.mainloop()
    root.destroy()
    
    # Click the selected coordinates after window is closed
    if selected_coords:
        time.sleep(0.05)
        touch_click(*selected_coords)
    
    return selected_coords

def get_coordinates(image_bytes, target_object):
    """Get coordinates using Moondream API."""
    api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI0YWQ5ODNkZC02NWIyLTRkNzMtYmMwNy1lMGQ0YWVkNWFjMzIiLCJpYXQiOjE3MzQwNTAwMzh9.GKoEaYT2_AjB6e9ZL3pczGygnWSjl7GKC08ZCJkaIVM'
    if not api_key:
        raise ValueError("Please set MOONDREAM_API_KEY environment variable")

    # Convert image to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # API endpoint
    url = "https://api.moondream.ai/v1/point"
    
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
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        print(f"API Response: {result}")
        
        if not result or 'points' not in result or not result['points']:
            raise ValueError(f"No coordinates found for '{target_object}'")
        
        points = result['points']
        
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        print(f"Screen size: {screen_width}x{screen_height}")
        
        # Convert all normalized coordinates to screen coordinates
        screen_points = []
        for point in points:
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

def take_screenshot():
    """Capture the current screen.
    
    Returns:
        bytes: Screenshot image data in PNG format
    
    The screenshot is taken of the entire screen and converted
    to bytes for API transmission.
    """
    screenshot = pyautogui.screenshot()
    # Convert to bytes
    img_byte_arr = BytesIO()
    screenshot.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def type_text(text):
    """Type text at current cursor position with a natural delay."""
    time.sleep(0.3)  # Reduced from 1.0 to 0.3 - just enough to let UI settle
    pyautogui.write(text, interval=0.02)  # Reduced from 0.05 to 0.02

def handle_action(search_term):
    """Handle different types of actions based on search term."""
    # Check for combined find and type command
    if ' and type ' in search_term.lower():
        # Split into find and type parts
        parts = search_term.lower().split(' and type ')
        find_part = parts[0]
        type_part = parts[1].strip()
        
        # Extract target object (remove 'find' if present)
        if find_part.startswith('find '):
            target = find_part[5:].strip()
        else:
            target = find_part.strip()
            
        # First find and click the target
        screenshot_bytes = take_screenshot()
        print(f"Looking for: {target}")
        
        points = get_coordinates(screenshot_bytes, target)
        print(f"Found {len(points)} possible locations for {target}")
        
        selected = visualize_coordinates(points)
        if selected:
            print(f"Clicked at coordinates: {selected}")
            # Now type the text
            type_text(type_part)
            return True
            
        return False

    # Check if it's a typing command
    if search_term.lower().startswith('type '):
        # Extract the text to type (everything after "type ")
        text_to_type = search_term[5:].strip()
        
        # First find a text box
        screenshot_bytes = take_screenshot()
        print("Looking for text box...")
        
        points = get_coordinates(screenshot_bytes, "text box")
        print(f"Found {len(points)} possible text box locations")
        
        # Show and select text box location
        selected = visualize_coordinates(points)
        if selected:
            print(f"Selected text box at: {selected}")
            touch_click(*selected)
            type_text(text_to_type)
            return True
            
        return False
    
    # Check if it's a find command
    if search_term.lower().startswith('find '):
        # Extract what to find (everything after "find ")
        target = search_term[5:].strip()
    else:
        # If "find" appears anywhere in the text, use everything after it
        find_idx = search_term.lower().find('find ')
        if find_idx >= 0:
            target = search_term[find_idx + 5:].strip()
        else:
            target = search_term
    
    # Default behavior - find and click
    screenshot_bytes = take_screenshot()
    print(f"Looking for: {target}")
    
    points = get_coordinates(screenshot_bytes, target)
    print(f"Found {len(points)} possible locations for {target}")
    
    selected = visualize_coordinates(points)
    if selected:
        print(f"Clicked at coordinates: {selected}")
        return True
    
    return False

class SearchWindow:
    """Modern Vercel-style command palette."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("")
        
        # Make window float on top and remove decorations
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)
        self.root.lift()
        
        # Set up modern dark theme colors
        self.bg_color = '#1C1C1C'  # Dark background
        self.text_color = '#FFFFFF'  # White text
        self.placeholder_color = '#6E7681'  # Gray placeholder
        self.border_color = '#333333'  # Subtle border
        self.highlight_color = '#2F81F7'  # Blue highlight
        
        # Center the window
        window_width = 800
        window_height = 80  # Slightly shorter for better proportions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/3)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Configure root background and border
        self.root.configure(bg=self.bg_color)
        
        # Create main container with padding and shadow effect
        self.container = tk.Frame(
            self.root,
            bg=self.bg_color,
            highlightbackground=self.border_color,
            highlightthickness=1,
        )
        
        # Add drop shadow
        shadow_frames = []
        shadow_size = 5
        shadow_color = '#000000'
        
        for i in range(shadow_size):
            alpha = 0.1 - (i * 0.02)
            shadow = tk.Frame(
                self.root,
                bg=shadow_color,
                height=2,
            )
            shadow.place(x=0, y=shadow_size+i, relwidth=1)
            shadow.lift()
            shadow_frames.append(shadow)
        
        self.container.pack(fill='both', expand=True, padx=1, pady=1)
        
        # Search icon (⌘)
        self.icon_label = tk.Label(
            self.container,
            text="⌘",
            font=('Segoe UI', 14),  # Slightly smaller icon
            fg=self.placeholder_color,
            bg=self.bg_color
        )
        self.icon_label.pack(side='left', padx=(15, 5), pady=0)  # Adjusted padding
        
        # Custom Entry widget
        self.search_var = tk.StringVar()
        self.entry = tk.Entry(
            self.container,
            textvariable=self.search_var,
            font=('Segoe UI', 16),
            fg=self.text_color,
            bg=self.bg_color,
            insertbackground=self.text_color,
            relief='flat',
            highlightthickness=0,
            bd=0
        )
        self.entry.pack(fill='x', expand=True, padx=(5, 15), pady=(20, 20))  # Centered vertically
        
        # Add placeholder
        self.entry.insert(0, "Type to find anything...")
        self.entry.config(fg=self.placeholder_color)
        
        # Bind events
        self.entry.bind('<FocusIn>', self.on_entry_click)
        self.entry.bind('<FocusOut>', self.on_focus_out)
        self.root.bind('<Return>', lambda e: self.submit())
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        
        # Add shortcut hint
        self.shortcut_label = tk.Label(
            self.container,
            text="alt+m to open, esc to close",
            font=('Segoe UI', 9),
            fg=self.placeholder_color,
            bg=self.bg_color
        )
        self.shortcut_label.pack(side='right', padx=(0, 15), pady=0)
        
        self.result = None
        
        # Ensure focus
        self.root.update_idletasks()
        self.root.lift()
        self.entry.focus_force()
        
        # Click to focus - faster initial display
        def click_entry():
            x = center_x + window_width//2
            y = center_y + window_height//2
            pyautogui.click(x, y)
            self.entry.focus_force()
        
        self.root.after(50, click_entry)  # Reduced from 100 to 50ms
        
        # Keep focused - check more frequently
        def ensure_focus():
            if not self.root.focus_get():
                self.root.lift()
                self.entry.focus_force()
            self.root.after(25, ensure_focus)  # Reduced from 50 to 25ms
        
        self.root.after(1, ensure_focus)
    
    def on_entry_click(self, event):
        """Handle entry field click."""
        if self.entry.get() == "Type to find anything...":
            self.entry.delete(0, tk.END)
            self.entry.config(fg=self.text_color)
    
    def on_focus_out(self, event):
        """Handle focus out."""
        if not self.entry.get():
            self.entry.insert(0, "Type to find anything...")
            self.entry.config(fg=self.placeholder_color)
    
    def submit(self):
        """Handle submission."""
        text = self.search_var.get()
        if text and text != "Type to find anything...":
            self.result = text
            self.root.destroy()
    
    def get_search_term(self):
        """Run the window and return the search term."""
        self.root.mainloop()
        return self.result

class VoiceSearchWindow:
    """Voice-controlled command palette for accessibility."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("")
        
        # Use same styling as text SearchWindow for consistency
        self.bg_color = '#1C1C1C'
        self.text_color = '#FFFFFF'
        self.placeholder_color = '#6E7681'
        self.border_color = '#333333'
        self.highlight_color = '#2F81F7'
        
        # Configure window
        window_width = 800
        window_height = 120  # Slightly taller to show more feedback
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/3)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Setup window attributes
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)
        self.root.configure(bg=self.bg_color)
        
        # Create container
        self.container = tk.Frame(
            self.root,
            bg=self.bg_color,
            highlightbackground=self.border_color,
            highlightthickness=1,
        )
        self.container.pack(fill='both', expand=True, padx=1, pady=1)
        
        # Microphone icon
        self.icon_label = tk.Label(
            self.container,
            text="🎤",
            font=('Segoe UI', 14),
            fg=self.text_color,
            bg=self.bg_color
        )
        self.icon_label.pack(side='left', padx=(15, 5), pady=0)
        
        # Status label
        self.status_label = tk.Label(
            self.container,
            text="Listening...",
            font=('Segoe UI', 16),
            fg=self.text_color,
            bg=self.bg_color
        )
        self.status_label.pack(fill='x', expand=True, padx=(5, 15), pady=(10, 0))
        
        # Recognized text label
        self.text_label = tk.Label(
            self.container,
            text="",
            font=('Segoe UI', 12),
            fg=self.placeholder_color,
            bg=self.bg_color,
            wraplength=window_width-60
        )
        self.text_label.pack(fill='x', expand=True, padx=(5, 15), pady=(0, 10))
        
        # Shortcut hint
        self.shortcut_label = tk.Label(
            self.container,
            text="esc to cancel",
            font=('Segoe UI', 9),
            fg=self.placeholder_color,
            bg=self.bg_color
        )
        self.shortcut_label.pack(side='right', padx=(0, 15), pady=0)
        
        # Voice recognition setup
        self.result = None
        self.voice_queue = queue.Queue()
        self.running = True
        
        # Add loading spinner frames
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = 0
        
        # Bind escape
        self.root.bind('<Escape>', lambda e: self.stop_recognition())
        
        # Start recognition in separate thread
        self.recognition_thread = threading.Thread(target=self.recognize_speech)
        self.recognition_thread.start()
        
        # Update UI periodically
        self.root.after(100, self.check_voice_queue)
    
    def recognize_speech(self):
        """Run voice recognition in background thread."""
        # Try enhanced model first, fall back to default if not available
        model_path = "vosk-model-en-us-0.22"
        try:
            if os.path.exists(model_path):
                print("Using enhanced voice recognition model...")
                model = vosk.Model(model_path)
            else:
                print("Using default voice recognition model...")
                model = vosk.Model(lang="en-us")
        except Exception as e:
            print(f"Error loading voice model: {str(e)}")
            print("Falling back to default voice recognition model...")
            try:
                model = vosk.Model(lang="en-us")
            except Exception as e:
                print(f"Failed to load any voice model: {str(e)}")
                self.running = False
                return
        
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000
            )
            
            recognizer = vosk.KaldiRecognizer(model, 16000)
            
            while self.running:
                data = stream.read(4000, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if result["text"]:
                        self.voice_queue.put(result["text"])
        
        except Exception as e:
            print(f"Error during voice recognition: {str(e)}")
            self.running = False
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if 'audio' in locals():
                audio.terminate()
    
    def animate_loading(self, text, frames=6):
        """Animate a loading spinner next to text."""
        for _ in range(frames):
            spinner = self.spinner_frames[self.spinner_idx]
            self.text_label.config(text=f"{text} {spinner}")
            self.root.update()
            self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)
            time.sleep(0.05)  # 50ms per frame
    
    def check_voice_queue(self):
        """Check for new voice recognition results."""
        try:
            while True:
                text = self.voice_queue.get_nowait()
                # Always update the display text first
                self.text_label.config(text=text)
                self.root.update()  # Force update the display
                
                # Then check for commands
                if "find" in text.lower() or "type" in text.lower():
                    # Show loading animation
                    self.animate_loading(text)
                    self.result = text
                    self.stop_recognition()
                    return
                    
        except queue.Empty:
            pass
        
        if self.running:
            self.root.after(100, self.check_voice_queue)
    
    def stop_recognition(self):
        """Stop voice recognition and close window."""
        self.running = False
        self.root.destroy()
    
    def get_search_term(self):
        """Run the window and return the recognized command."""
        self.root.mainloop()
        return self.result

def main():
    """Main function handling the program flow."""
    print("\nComputer Vision-Based Object Location Tool")
    print("----------------------------------------")
    print("Voice Recognition:")
    
    model_path = "vosk-model-en-us-0.22"
    if os.path.exists(model_path):
        print("✓ Using enhanced voice recognition model")
    else:
        print("! Using default voice recognition model")
        print("\nTo install enhanced model (recommended), download using either:")
        print("wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip")
        print("curl -LO https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip")
        print("\nThen extract to 'vosk-model-en-us-0.22' folder in script directory:")
        print("unzip vosk-model-en-us-0.22.zip")
    
    print("\nControls:")
    print("- Press 'Alt+M' for text search")
    print("- Press 'Alt+N' for voice control")
    print("- Press Ctrl+C to exit\n")

    running = True
    while running:
        try:
            # Wait for either hotkey
            if keyboard.is_pressed('alt+m'):
                # Text interface
                keyboard.wait('alt+m', suppress=True)  # Wait for release to prevent double triggers
                search_window = SearchWindow()
                search_term = search_window.get_search_term()
            elif keyboard.is_pressed('alt+n'):
                # Voice interface
                keyboard.wait('alt+n', suppress=True)  # Wait for release to prevent double triggers
                voice_window = VoiceSearchWindow()
                search_term = voice_window.get_search_term()
            else:
                # No hotkey pressed, continue checking
                time.sleep(0.1)
                continue
            
            if not search_term:
                continue
            
            time.sleep(0.05)
            handle_action(search_term)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            running = False
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            if isinstance(e, KeyboardInterrupt):
                running = False

if __name__ == "__main__":
    main()
