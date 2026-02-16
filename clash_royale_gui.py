# clash_royale_gui.py
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from typing import List, Dict
import os
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests
import json
from collections import Counter
import re
from urllib.parse import unquote
import pickle
import training_data
import cards
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import your existing classes and functions
from clash_royale_archetype_classifier import *


class DragDropCardGUI:
    def __init__(self, parent_frame, on_deck_update_callback):
        self.parent = parent_frame
        self.on_deck_update = on_deck_update_callback
        self.card_images = {}
        self.card_buttons = {}
        self.selected_cards = []
        self.deck_slots = []

        self.setup_card_database()
        self.create_drag_drop_interface()
        self.selected_cards = []

    

    def setup_card_database(self):
        """Setup card database and load images"""
        self.all_cards = []
        for card_id, card_info in CLASH_ROYALE_CARDS.items():
            self.all_cards.append({
                'id': card_id,
                'name': card_info['name'],
                'elixir': card_info['elixir'],
                'type': card_info['type'],
                'rarity': card_info['rarity']
            })

        # Sort cards by elixir cost then name
        self.all_cards.sort(key=lambda x: (x['elixir'], x['name']))

    def create_drag_drop_interface(self):
        """Create the drag and drop interface"""
        # Main container for drag-drop
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left side - All cards
        left_frame = ttk.LabelFrame(main_frame, text="Available Cards", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Search and filter
        self.create_search_filter(left_frame)

        # Cards scrollable area
        self.create_cards_scrollable(left_frame)

        # Right side - Deck builder
        right_frame = ttk.LabelFrame(main_frame, text="Your Deck (0/8)", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.deck_label = right_frame  # Store reference to update title

        # Deck slots
        self.create_deck_slots(right_frame)

        # Deck controls
        self.create_deck_controls(right_frame)


    def create_search_filter(self, parent):
        """Create search and filter controls"""
        search_frame = ttk.Frame(parent)
        search_frame.pack(fill=tk.X, pady=(0, 10))

        # Search entry
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        self.search_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.search_entry.bind('<KeyRelease>', self.on_search)

        # Elixir filter
        ttk.Label(search_frame, text="Elixir:").pack(side=tk.LEFT, padx=(0, 5))
        self.elixir_var = tk.StringVar(value="All")
        elixir_combo = ttk.Combobox(search_frame, textvariable=self.elixir_var,
                                    values=["All", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                    state="readonly", width=8)
        elixir_combo.pack(side=tk.LEFT, padx=(0, 10))
        elixir_combo.bind('<<ComboboxSelected>>', self.on_filter)

        # Type filter
        ttk.Label(search_frame, text="Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.type_var = tk.StringVar(value="All")
        type_combo = ttk.Combobox(search_frame, textvariable=self.type_var,
                                  values=["All", "Troop", "Spell", "Building"],
                                  state="readonly", width=10)
        type_combo.pack(side=tk.LEFT, padx=(0, 10))
        type_combo.bind('<<ComboboxSelected>>', self.on_filter)

        # Rarity filter (new)
        ttk.Label(search_frame, text="Rarity:").pack(side=tk.LEFT, padx=(0, 5))
        self.rarity_var = tk.StringVar(value="All")
        rarity_combo = ttk.Combobox(search_frame, textvariable=self.rarity_var,
                                    values=["All", "Common", "Rare", "Epic", "Legendary", "Champion"],
                                    state="readonly", width=12)
        rarity_combo.pack(side=tk.LEFT)
        rarity_combo.bind('<<ComboboxSelected>>', self.on_filter)

    def apply_filters(self):
        """Apply search and filters to card display"""
        search_text = self.search_var.get().lower()
        elixir_filter = self.elixir_var.get()
        type_filter = self.type_var.get()
        rarity_filter = getattr(self, 'rarity_var', tk.StringVar(value="All")).get()

        # Map rarity names to numeric values used in card data (fallback if card stores strings)
        rarity_name_to_value = {
            "Common": 1,
            "Rare": 2,
            "Epic": 3,
            "Legendary": 4,
            "Champion": 5
        }

        filtered_cards = []

        for card in self.all_cards:
            # Search filter
            if search_text and search_text not in card['name'].lower():
                continue

            # Elixir filter
            if elixir_filter != "All" and str(card['elixir']) != elixir_filter:
                continue

            # Type filter
            if type_filter != "All" and card['type'].title() != type_filter:
                continue

            # Rarity filter
            if rarity_filter != "All":
                target = rarity_name_to_value.get(rarity_filter)
                if target is not None:
                    # card['rarity'] is typically numeric
                    try:
                        if int(card['rarity']) != target:
                            continue
                    except Exception:
                        # fallback to string comparison if stored as text
                        if str(card['rarity']).title() != rarity_filter:
                            continue
                else:
                    # fallback string compare
                    if str(card.get('rarity', '')).title() != rarity_filter:
                        continue

            filtered_cards.append(card)

        # Display and ensure list is reset to top so search never leaves you halfway
        self.display_cards(filtered_cards)
        try:
            self.canvas.yview_moveto(0)
        except Exception:
            pass

    # python
    def create_cards_scrollable(self, parent):
        """Create scrollable area for cards"""
        # Create frame with scrollbar
        card_container = ttk.Frame(parent)
        card_container.pack(fill=tk.BOTH, expand=True)

        # Canvas and scrollbar
        self.canvas = tk.Canvas(card_container, bg='#34495e')
        scrollbar = ttk.Scrollbar(card_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse enter/leave so wheel works only when pointer is over the card list
        self.scrollable_frame.bind("<Enter>", lambda e: self._bind_to_mousewheel())
        self.scrollable_frame.bind("<Leave>", lambda e: self._unbind_from_mousewheel())

        # Load and display cards
        self.load_card_images()
        self.display_cards()



    # Helper methods for mouse-wheel scrolling
    def _bind_to_mousewheel(self):
        """Bind mousewheel events when cursor enters the card area"""
        try:
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)  # Windows / Mac
            self.canvas.bind_all("<Button-4>", self._on_mousewheel)  # Linux scroll up
            self.canvas.bind_all("<Button-5>", self._on_mousewheel)  # Linux scroll down
        except Exception:
            pass

    def _unbind_from_mousewheel(self):
        """Unbind mousewheel events when cursor leaves the card area"""
        try:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        except Exception:
            pass

    def _on_mousewheel(self, event):
        """Handle mouse wheel scroll across platforms"""
        try:
            # Linux (Button-4/5)
            if hasattr(event, "num") and event.num in (4, 5):
                if event.num == 4:
                    self.canvas.yview_scroll(-1, "units")
                else:
                    self.canvas.yview_scroll(1, "units")
                return

            # Windows / macOS - event.delta present
            delta = 0
            if hasattr(event, "delta"):
                # On Windows, delta is multiple of 120 (positive when scrolling up)
                delta = int(-1 * (event.delta / 120))
            if delta == 0:
                # fallback single unit
                delta = 1 if getattr(event, "delta", -1) < 0 else -1

            self.canvas.yview_scroll(delta, "units")
        except Exception:
            pass

    def load_card_images(self):
        """Load card images from images directory"""
        images_dir = "images"  # Change this to your images directory path

        if not os.path.exists(images_dir):
            print(f"Warning: Images directory '{images_dir}' not found. Using text buttons.")
            return

        for card in self.all_cards:
            card_name = card['name']
            image_path = os.path.join(images_dir, f"{card_name}.png")

            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                    image = image.resize((80, 100), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    self.card_images[card_name] = photo
                except Exception as e:
                    print(f"Error loading image for {card_name}: {e}")
                    self.card_images[card_name] = None
            else:
                self.card_images[card_name] = None

    def display_cards(self, cards_to_show=None):
        """Display cards in the scrollable area"""
        if cards_to_show is None:
            cards_to_show = self.all_cards

        # Clear existing cards
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.card_buttons = {}

        # Display cards in grid
        row, col = 0, 0
        max_cols = 6

        for card in cards_to_show:
            card_frame = ttk.Frame(self.scrollable_frame, relief='raised', borderwidth=1)
            card_frame.grid(row=row, column=col, padx=2, pady=2, sticky='nsew')
            card_frame.card_data = card  # Store card data

            # Make frame responsive to drag
            card_frame.bind('<Button-1>', self.on_card_click)

            if card['name'] in self.card_images and self.card_images[card['name']]:
                # Use image button
                btn = tk.Label(card_frame, image=self.card_images[card['name']],
                               cursor='hand2', bg='#2c3e50')
                btn.pack(padx=2, pady=2)
                btn.card_data = card
                btn.bind('<Button-1>', self.on_card_click)
            else:
                # Use text button
                btn_text = f"{card['name']}\n({card['elixir']}⏱️)"
                btn = tk.Label(card_frame, text=btn_text, cursor='hand2',
                               bg='#34495e', fg='white', wraplength=80, justify='center')
                btn.pack(padx=5, pady=5)
                btn.card_data = card
                btn.bind('<Button-1>', self.on_card_click)

            self.card_buttons[card['name']] = btn

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Configure grid weights
        for i in range(max_cols):
            self.scrollable_frame.columnconfigure(i, weight=1)

    def create_deck_slots(self, parent):
        """Create deck slot areas"""
        slots_frame = ttk.Frame(parent)
        slots_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create 2 rows of 4 slots
        for row in range(2):
            for col in range(4):
                slot_num = row * 4 + col
                slot_frame = ttk.Frame(slots_frame, relief='sunken', borderwidth=2,
                                       width=100, height=120)
                slot_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
                slot_frame.grid_propagate(False)  # Keep fixed size

                # Slot label
                slot_label = tk.Label(slot_frame, text=f"Slot {slot_num + 1}",
                                      bg='#95a5a6', fg='black')
                slot_label.pack(fill=tk.BOTH, expand=True)

                # Store slot info
                slot_info = {
                    'frame': slot_frame,
                    'label': slot_label,
                    'card': None
                }
                self.deck_slots.append(slot_info)

                # Make droppable
                slot_frame.bind('<Button-1>', lambda e, s=slot_num: self.on_slot_click(s))

        # Configure grid weights
        for i in range(4):
            slots_frame.columnconfigure(i, weight=1)
        for i in range(2):
            slots_frame.rowconfigure(i, weight=1)

    def create_deck_controls(self, parent):
        """Create deck control buttons"""
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X)

        ttk.Button(controls_frame, text="Clear Deck",
                   command=self.clear_deck).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(controls_frame, text="Auto-fill Example",
                   command=self.auto_fill_example).pack(side=tk.LEFT, padx=(0, 5))

        # self.deck_status = ttk.Label(controls_frame, text="Deck: 0/8 cards")
        # self.deck_status.pack(side=tk.RIGHT)

    # python
    # Replace these methods inside class DragDropCardGUI in `clash_royale_gui.py`

    def on_card_click(self, event):
        """Handle card click - add to first empty deck slot, but prevent duplicates"""
        card_data = event.widget.card_data

        # Prevent duplicates: check existing slots by id or name
        for slot in self.deck_slots:
            existing = slot.get('card')
            if existing is not None:
                if ('id' in existing and 'id' in card_data and existing['id'] == card_data['id']) or \
                        (existing.get('name') == card_data.get('name')):
                    messagebox.showinfo("Duplicate Card", f"'{card_data.get('name')}' is already in your deck.")
                    return

        # Find first empty slot
        empty_slot = None
        for slot in self.deck_slots:
            if slot['card'] is None:
                empty_slot = slot
                break

        if empty_slot:
            self.add_card_to_slot(card_data, empty_slot)
        else:
            messagebox.showinfo("Deck Full", "Your deck is full! Remove a card first.")

    def on_slot_click(self, slot_index):
        """Handle slot click - remove card from slot"""
        slot = self.deck_slots[slot_index]
        if slot['card'] is not None:
            self.remove_card_from_slot(slot)


    def add_card_to_slot(self, card_data, slot):
        """Add card to a deck slot, guarding against duplicates"""
        # Extra guard in case other code calls this directly
        for s in self.deck_slots:
            existing = s.get('card')
            if existing is not None:
                if ('id' in existing and 'id' in card_data and existing['id'] == card_data['id']) or \
                        (existing.get('name') == card_data.get('name')):
                    messagebox.showinfo("Duplicate Card", f"'{card_data.get('name')}' is already in your deck.")
                    return

        slot['card'] = card_data

        # Clear slot UI
        for widget in slot['frame'].winfo_children():
            widget.destroy()

        # Add card to slot (image or text)
        if card_data['name'] in self.card_images and self.card_images[card_data['name']]:
            card_label = tk.Label(slot['frame'], image=self.card_images[card_data['name']],
                                  bg='#27ae60')
            card_label.pack(fill=tk.BOTH, expand=True)
        else:
            card_text = f"{card_data['name']}\n({card_data['elixir']}⏱️)"
            card_label = tk.Label(slot['frame'], text=card_text, bg='#27ae60',
                                  fg='white', wraplength=80, justify='center')
            card_label.pack(fill=tk.BOTH, expand=True)

        card_label.card_data = card_data
        card_label.bind('<Button-1>', lambda e: self.remove_card_from_slot(slot))

        self.update_deck_status()



    def remove_card_from_slot(self, slot):
        """Remove card from deck slot"""
        slot['card'] = None

        # Clear and reset slot
        for widget in slot['frame'].winfo_children():
            widget.destroy()

        slot_label = tk.Label(slot['frame'], text=f"Slot {self.deck_slots.index(slot) + 1}",
                              bg='#95a5a6', fg='black')
        slot_label.pack(fill=tk.BOTH, expand=True)

        self.update_deck_status()

    # python
    def update_deck_status(self):
        """Update deck status and show running total and average elixir."""
        current_cards = [slot['card'] for slot in self.deck_slots if slot['card'] is not None]
        card_count = len(current_cards)

        # Compute total and average elixir safely (handle missing or non-dict entries)
        total_elixir = sum((c.get('elixir', 0) if isinstance(c, dict) else 0) for c in current_cards)
        avg_elixir = (total_elixir / card_count) if card_count > 0 else 0.0

        # Update the deck frame title with elixir info (self.deck_label is the LabelFrame)
        try:
            self.deck_label.configure(
                text=f"Your Deck ({card_count}/8)  —  Avg: {avg_elixir:.2f}  •  Total: {total_elixir}")
        except Exception:
            pass

        # Notify parent about deck update
        if self.on_deck_update:
            self.on_deck_update(current_cards)

    def clear_deck(self):
        """Clear all cards from deck"""
        for slot in self.deck_slots:
            if slot['card'] is not None:
                self.remove_card_from_slot(slot)

    def auto_fill_example(self):
        """Auto-fill with an example deck"""
        example_cards = [
            "MegaKnight", "PEKKA", "Bandit", "Ghost",
            "EWiz", "Zap", "Poison", "Tornado"
        ]

        self.clear_deck()

        for i, card_name in enumerate(example_cards):
            if i < len(self.deck_slots):
                # Find card data
                card_data = None
                for card in self.all_cards:
                    if card['name'].lower() == card_name.lower():
                        card_data = card
                        break

                if card_data:
                    self.add_card_to_slot(card_data, self.deck_slots[i])

    def on_search(self, event=None):
        """Handle search filter"""
        self.apply_filters()

    def on_filter(self, event=None):
        """Handle filter changes"""
        self.apply_filters()



    def get_deck_card_names(self):
        """Get list of card names in current deck"""
        return [slot['card']['name'] for slot in self.deck_slots if slot['card'] is not None]


class ClashRoyaleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Clash Royale Archetype Classifier")
        self.root.geometry("1200x800")  # Larger window for drag-drop
        self.root.configure(bg='#2c3e50')

        # Initialize predictor
        self.predictor = None
        self.load_model_attempted = False
        self.current_deck_card_ids = []  # Add this line to track current deck

        # Style configuration
        self.setup_styles()

        # Create main interface
        self.create_main_interface()

        # Try to load model in background
        self.load_model_background()

    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        style.configure('Title.TLabel',
                        background='#2c3e50',
                        foreground='#ecf0f1',
                        font=('Arial', 16, 'bold'))

        style.configure('Subtitle.TLabel',
                        background='#2c3e50',
                        foreground='#bdc3c7',
                        font=('Arial', 12))

        style.configure('Card.TFrame',
                        background='#34495e',
                        relief='raised',
                        borderwidth=1)

        style.configure('Accent.TButton',
                        background='#e74c3c',
                        foreground='white',
                        font=('Arial', 10, 'bold'))

        style.configure('Success.TButton',
                        background='#27ae60',
                        foreground='white',
                        font=('Arial', 10, 'bold'))

    # python
    def create_main_interface(self):
        """Create the main GUI interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.create_drag_drop_tab()
        self.create_text_input_tab()
        self.create_results_tab()
        self.create_deck_overview_tab()

        # Make Text Input tab active and show the card names input by default
        try:
            self.notebook.select(0)  # select the Text Input tab (0-based index)
            self.on_method_change()  # ensure the names_frame is displayed
        except Exception:
            pass



   # ...existing code...
    def create_deck_overview_tab(self):
        """Create Deck overview tab with pie (left) + bar (center) + pie (right) charts and deck images"""
        overview_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(overview_frame, text="Deck overview")

        # Chart area (pie left, bar center, pie right)
        chart_frame = ttk.LabelFrame(overview_frame, text="Deck Summary", padding="8")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 8))

        # Create matplotlib figure with three subplots
        fig = plt.Figure(figsize=(12, 3.5), dpi=100)
        ax_pie = fig.add_subplot(1, 3, 1)
        ax_bar = fig.add_subplot(1, 3, 2)
        ax_train_pie = fig.add_subplot(1, 3, 3)

        # initial placeholders
        ax_pie.text(0.5, 0.5, "No deck", ha='center', va='center')
        ax_bar.text(0.5, 0.5, "No probabilities", ha='center', va='center')
        ax_train_pie.text(0.5, 0.5, "No training data", ha='center', va='center')
        ax_pie.set_axis_off()
        ax_bar.set_axis_off()
        ax_train_pie.set_axis_off()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.overview_fig = fig
        self.overview_ax_pie = ax_pie
        self.overview_ax_bar = ax_bar
        self.overview_ax_train_pie = ax_train_pie
        self.overview_canvas = canvas
        self.overview_canvas_widget = canvas_widget

        # Images area (deck)
        images_frame_label = ttk.LabelFrame(overview_frame, text="Deck", padding="8")
        images_frame_label.pack(fill=tk.X, padx=5, pady=(0, 8))

        images_frame = ttk.Frame(images_frame_label)
        images_frame.pack(fill=tk.X, expand=False)
        self.overview_images_frame = images_frame
        self.overview_card_photos = []  # keep references to PhotoImage objects

        # ...existing code...
    def update_deck_overview(self, all_probabilities: dict, deck_card_ids: list):
        """Update overview tab with:
        - pie chart (left) showing Troop/Spell/Building ratios
        - bar chart (center) showing archetype probabilities
        - pie chart (right) showing archetype ratios in training data (from training_data.py)
        - stats (avg elixir, 4-card cycle) as suptitle and deck images below
        """
        try:
            # Compute simple deck stats (avg elixir, 4-card cycle)
            avg_elixir = None
            four_card_cycle = None
            try:
                stats = calculate_deck_stats(deck_card_ids)
                avg_elixir = stats.get('average_elixir')
                four_card_cycle = stats.get('four_card_cycle')
            except Exception:
                elixirs = []
                for cid in deck_card_ids or []:
                    try:
                        info = get_card_info(cid)
                        el = info.get('elixir', 0) or 0
                        elixirs.append(float(el))
                    except Exception:
                        continue
                if elixirs:
                    avg_elixir = sum(elixirs) / len(elixirs)
                    four_card_cycle = avg_elixir * 4

            # Prepare figure and axes
            fig = getattr(self, 'overview_fig', None)
            ax_pie = getattr(self, 'overview_ax_pie', None)
            ax_bar = getattr(self, 'overview_ax_bar', None)
            ax_train_pie = getattr(self, 'overview_ax_train_pie', None)
            if fig is None or ax_pie is None or ax_bar is None or ax_train_pie is None:
                self.create_deck_overview_tab()
                fig = self.overview_fig
                ax_pie = self.overview_ax_pie
                ax_bar = self.overview_ax_bar
                ax_train_pie = self.overview_ax_train_pie

            ax_pie.clear()
            ax_bar.clear()
            ax_train_pie.clear()

            # Title/stats above charts
            stats_parts = []
            if avg_elixir is not None:
                stats_parts.append(f"Avg Elixir: {avg_elixir:.2f}")
            if four_card_cycle is not None:
                stats_parts.append(f"4-card cycle: {four_card_cycle:.2f}")
            stats_title = "  •  ".join(stats_parts) if stats_parts else "No deck stats available"
            try:
                fig.suptitle(stats_title, fontsize=10, y=0.98)
            except Exception:
                pass

            # PIE CHART (LEFT): Troop / Spell / Building (and Other)
            type_counts = {"troop": 0, "spell": 0, "building": 0, "other": 0}
            for cid in deck_card_ids or []:
                try:
                    info = get_card_info(cid)
                    ctype = (info.get('type') or '').lower()
                    if 'troop' in ctype or 'unit' in ctype or ctype == 'card' or ctype == 'troops':
                        key = 'troop'
                    elif 'spell' in ctype or ctype == 'spells':
                        key = 'spell'
                    elif 'building' in ctype or ctype == 'buildings':
                        key = 'building'
                    else:
                        key = 'other'
                    type_counts[key] += 1
                except Exception:
                    type_counts['other'] += 1

            labels = []
            sizes = []
            colors = ['#2ecc71', '#3498db', '#f1c40f', '#95a5a6']  # troop, spell, building, other
            for k, color in zip(['troop', 'spell', 'building', 'other'], colors):
                if type_counts[k] > 0:
                    labels.append(f"{k.title()} ({type_counts[k]})")
                    sizes.append(type_counts[k])

            if not sizes:
                ax_pie.text(0.5, 0.5, "No cards", ha='center', va='center')
                ax_pie.set_axis_off()
            else:
                ax_pie.pie(sizes, labels=labels, autopct='%1.0f%%', colors=[c for c in colors[:len(sizes)]], startangle=90)
                ax_pie.axis('equal')
                ax_pie.set_title("Card Type Ratio")

            # BAR CHART (CENTER): Archetype probabilities
            if not all_probabilities:
                ax_bar.text(0.5, 0.5, "No probabilities available", ha='center', va='center')
                ax_bar.set_axis_off()
            else:
                items = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
                labels_bar = [k for k, _ in items]
                values = [v for _, v in items]

                x_pos = list(range(len(labels_bar)))
                bars = ax_bar.bar(x_pos, values, color='#34495e')

                ax_bar.set_xticks(x_pos)
                ax_bar.set_xticklabels(labels_bar, rotation=45, ha='right')
                ax_bar.set_ylabel("Probability")
                ax_bar.set_ylim(0, 1.0)

                for rect, v in zip(bars, values):
                    height = rect.get_height()
                    ax_bar.text(rect.get_x() + rect.get_width() / 2, height + 0.01, f"{v:.1%}", ha='center', va='bottom', fontsize=8)

            # TRAINING DATA PIE (RIGHT): ratios of archetypes present in training_data.py
            train_counts = {}
            try:
                # Primary source: training_data.training_data list of dicts with 'archetype'
                from training_data import training_data as td
                for entry in td:
                    if not isinstance(entry, dict):
                        continue
                    a = entry.get('archetype')
                    if a:
                        train_counts[a] = train_counts.get(a, 0) + 1
            except Exception:
                train_counts = {}

            # fallback: try predictor/trainer/processor attributes or archetype jsons if training_data not present
            if not train_counts:
                try:
                    proc = getattr(getattr(self, 'predictor', None), 'trainer', None)
                    proc = getattr(proc, 'processor', proc)
                    counts_attr = None
                    if proc is not None:
                        for key in ('archetype_counts', 'archetype_distribution', 'archetype_freq', 'archetype_freqs'):
                            if hasattr(proc, key):
                                counts_attr = getattr(proc, key)
                                break
                    if isinstance(counts_attr, dict):
                        train_counts = counts_attr.copy()
                    elif isinstance(counts_attr, (list, tuple)) and getattr(proc, 'archetypes', None):
                        for name, cnt in zip(proc.archetypes, counts_attr):
                            train_counts[name] = float(cnt)
                    elif getattr(proc, 'archetypes', None):
                        for a in proc.archetypes:
                            train_counts[a] = train_counts.get(a, 0) + 1.0
                except Exception:
                    pass

            # fallback from archetype json files
            if not train_counts:
                try:
                    
                    for path in glob.glob(os.path.join(os.path.dirname(__file__), "*archetype*.json")):
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                f.close()
                            if isinstance(data, dict):
                                candidate = {}
                                for k, v in data.items():
                                    if isinstance(v, (int, float)):
                                        candidate[k] = float(v)
                                    elif isinstance(v, list):
                                        candidate[k] = float(len(v))
                                if candidate:
                                    train_counts = candidate
                                    break
                            
                        except Exception:
                            continue
                except Exception:
                    pass

            # Normalize and draw training pie
            

            if not train_counts:
                ax_train_pie.text(0.5, 0.5, "No training data", ha='center', va='center')
                ax_train_pie.set_axis_off()
            else:
                labels_t = list(train_counts.keys())
                sizes_t = [float(train_counts[k]) for k in labels_t]
                pairs = sorted(zip(labels_t, sizes_t), key=lambda x: x[1], reverse=True)
                labels_t, sizes_t = zip(*pairs)
                labels_t_disp = [l if len(l) <= 18 else l[:15] + "..." for l in labels_t]
                cmap = plt.get_cmap('tab20')
                colors_t = [cmap(i % 20) for i in range(len(labels_t))]
                ax_train_pie.pie(sizes_t, labels=labels_t_disp, autopct='%1.0f%%', colors=colors_t, startangle=90)
                ax_train_pie.axis('equal')
                ax_train_pie.set_title("Archetypes in Training Data")

            # Layout adjustments
            try:
                fig.tight_layout(rect=[0, 0, 1, 0.94])
            except Exception:
                fig.tight_layout()

            # Redraw canvas
            try:
                if hasattr(self, 'overview_canvas') and self.overview_canvas is not None:
                    self.overview_canvas.draw()
                else:
                    canvas = FigureCanvasTkAgg(fig, master=self.overview_canvas_widget.master if hasattr(self, 'overview_canvas_widget') else None)
                    self.overview_canvas = canvas
                    self.overview_canvas_widget = canvas.get_tk_widget()
                    self.overview_canvas_widget.pack(fill=tk.BOTH, expand=True)
                    self.overview_canvas.draw()
            except Exception:
                pass

            # Update deck images below chart
            for w in getattr(self, 'overview_images_frame', []).winfo_children():
                w.destroy()
            self.overview_card_photos.clear()

            images_dir = "images"
            #print("kinda good")
            for cid in deck_card_ids or []:

                try:
                    card_info = get_card_info(cid)
                    name = card_info.get('name', str(cid))
                    image_path = os.path.join(images_dir, f"{name}.png")
                    if not os.path.exists(image_path):
                        image_path = os.path.join(images_dir, f"{name}.jpg")
                    if os.path.exists(image_path):
                        im = Image.open(image_path).resize((80, 100), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(im)
                        lbl = tk.Label(self.overview_images_frame, image=photo, bg='#2c3e50')
                        lbl.pack(side=tk.LEFT, padx=4, pady=4)
                        lbl.card_id = cid
                        self.overview_card_photos.append(photo)
                    else:
                        lbl = tk.Label(self.overview_images_frame, text=name, bg='#34495e', fg='white', wraplength=80, justify='center')
                        lbl.pack(side=tk.LEFT, padx=4, pady=4)
                except Exception as e:
                    print("error loading overview image for card id", cid, ":", e)
                    lbl = tk.Label(self.overview_images_frame, text=str(cid), bg='#34495e', fg='white')
                    lbl.pack(side=tk.LEFT, padx=4, pady=4)

        except Exception as e:
            try:
                for w in getattr(self, 'overview_images_frame', []).winfo_children():
                    w.destroy()
            except Exception:
                pass
            lbl = ttk.Label(getattr(self, 'overview_images_frame', self.root), text=f"Failed to render overview: {e}")
            lbl.pack()
    # ...existing code...

    def create_drag_drop_tab(self):
        """Create drag and drop tab"""
        drag_drop_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(drag_drop_frame, text="Drag & Drop Builder")

        # Model status
        status_frame = ttk.Frame(drag_drop_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self.status_label = ttk.Label(status_frame,
                                      text="Loading model...",
                                      style='Subtitle.TLabel')
        self.status_label.pack(side=tk.LEFT)

        # Create drag-drop interface
        self.drag_drop_gui = DragDropCardGUI(drag_drop_frame, self.on_deck_update)

        # Predict button for drag-drop
        predict_frame = ttk.Frame(drag_drop_frame)
        predict_frame.pack(fill=tk.X, pady=(10, 0))

        self.drag_predict_btn = ttk.Button(predict_frame,
                                           text="Predict Archetype from Deck",
                                           command=self.predict_from_drag_drop,
                                           style='Accent.TButton',
                                           state='disabled')
        self.drag_predict_btn.pack()

    def create_text_input_tab(self):
        """Create traditional text input tab"""
        text_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(text_frame, text="Text Input")

        # Input method selection
        self.create_input_method_section(text_frame)

    def create_results_tab(self):
        """Create results display tab"""
        results_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(results_frame, text="Analysis Results")

        # Results text area
        self.create_results_section(results_frame)

        # Training section
        self.create_training_section(results_frame)

    def create_input_method_section(self, parent):
        """Create deck input method selection section"""
        input_frame = ttk.LabelFrame(parent, text="Text Input Methods", padding="15")
        input_frame.pack(fill=tk.BOTH, expand=True)

        # Method selection
        method_frame = ttk.Frame(input_frame)
        method_frame.pack(fill=tk.X, pady=(0, 15))

        self.input_method = tk.StringVar(value="names")

        ttk.Radiobutton(method_frame, text="Card Names",
                        variable=self.input_method, value="names",
                        command=self.on_method_change).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(method_frame, text="Deck URL",
                        variable=self.input_method, value="url",
                        command=self.on_method_change).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(method_frame, text="Card IDs",
                        variable=self.input_method, value="ids",
                        command=self.on_method_change).pack(side=tk.LEFT)

        # Input areas
        self.create_names_input(input_frame)
        self.create_url_input(input_frame)
        self.create_ids_input(input_frame)

        # Predict button
        self.predict_btn = ttk.Button(input_frame,
                                      text="Predict Archetype",
                                      command=self.predict_archetype,
                                      style='Accent.TButton',
                                      state='disabled')
        self.predict_btn.pack(pady=(10, 0))

    def create_names_input(self, parent):
        """Create card names input section"""
        self.names_frame = ttk.Frame(parent)

        instruction_label = ttk.Label(self.names_frame,
                                      text="Enter 8 card names (one per line):")
        instruction_label.pack(anchor=tk.W, pady=(0, 10))

        # Card entries frame
        entries_frame = ttk.Frame(self.names_frame)
        entries_frame.pack(fill=tk.X)

        self.card_entries = []
        for i in range(8):
            row_frame = ttk.Frame(entries_frame)
            row_frame.pack(fill=tk.X, pady=2)

            label = ttk.Label(row_frame, text=f"Card {i + 1}:", width=8)
            label.pack(side=tk.LEFT)

            entry = ttk.Entry(row_frame, width=30)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
            self.card_entries.append(entry)

            # Add autocomplete
            self.setup_autocomplete(entry)

        # Control buttons
        control_frame = ttk.Frame(self.names_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(control_frame, text="Fill Example Deck",
                   command=self.fill_example_deck).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(control_frame, text="Clear All Fields",
                   command=self.clear_text_fields).pack(side=tk.LEFT)

    def create_url_input(self, parent):
        """Create deck URL input section"""
        self.url_frame = ttk.Frame(parent)

        instruction_label = ttk.Label(self.url_frame,
                                      text="Enter Clash Royale deck URL:")
        instruction_label.pack(anchor=tk.W, pady=(0, 10))

        self.url_entry = ttk.Entry(self.url_frame, width=80)
        self.url_entry.pack(fill=tk.X)

        # Example URL
        example_label = ttk.Label(self.url_frame,
                                  text="Example: clashroyale://copyDeck?deck=26000063;26000015;...",
                                  font=('Arial', 8),
                                  foreground='#7f8c8d')
        example_label.pack(anchor=tk.W, pady=(5, 0))

        # Control buttons
        control_frame = ttk.Frame(self.url_frame)
        control_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(control_frame, text="Clear URL",
                   command=lambda: self.url_entry.delete(0, tk.END)).pack(side=tk.LEFT)

    def create_ids_input(self, parent):
        """Create card IDs input section"""
        self.ids_frame = ttk.Frame(parent)

        instruction_label = ttk.Label(self.ids_frame,
                                      text="Enter 8 card IDs (semicolon-separated):")
        instruction_label.pack(anchor=tk.W, pady=(0, 10))

        self.ids_entry = ttk.Entry(self.ids_frame, width=80)
        self.ids_entry.pack(fill=tk.X)

        # Example
        example_label = ttk.Label(self.ids_frame,
                                  text="Example: 26000063;26000015;26000009;26000018;26000068;28000012;28000015;27000007",
                                  font=('Arial', 8),
                                  foreground='#7f8c8d')
        example_label.pack(anchor=tk.W, pady=(5, 0))

        # Control buttons
        control_frame = ttk.Frame(self.ids_frame)
        control_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(control_frame, text="Clear IDs",
                   command=lambda: self.ids_entry.delete(0, tk.END)).pack(side=tk.LEFT)

    def create_results_section(self, parent):
        """Create results display section"""
        results_frame = ttk.LabelFrame(parent, text="Prediction Results", padding="15")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame,
                                                      height=20,
                                                      wrap=tk.WORD,
                                                      font=('Consolas', 10),
                                                      bg='#2c3e50',
                                                      fg='#ecf0f1',
                                                      insertbackground='white')
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Make text widget read-only
        self.results_text.config(state=tk.DISABLED)

    def create_training_section(self, parent):
        """Create model training section"""
        training_frame = ttk.Frame(parent)
        training_frame.pack(fill=tk.X)

        # Training button
        self.train_btn = ttk.Button(training_frame,
                                    text="Train New Model",
                                    command=self.train_model,
                                    style='Success.TButton')
        self.train_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Progress bar for training
        self.progress = ttk.Progressbar(training_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def setup_autocomplete(self, entry):
        """Setup autocomplete for card name entry"""

        def autocomplete(event):
            current_text = entry.get().lower()
            if not current_text:
                return

            matches = [name for name in CARD_NAME_TO_ID.keys()
                       if name.startswith(current_text)]

            if matches:
                entry.delete(0, tk.END)
                entry.insert(0, matches[0].title())
                entry.select_range(len(current_text), tk.END)

        entry.bind('<Tab>', autocomplete)

    def on_deck_update(self, current_cards):
        """Callback when drag-drop deck is updated"""
        # Enable predict button if deck is complete
        if len(current_cards) == 8:
            self.drag_predict_btn.config(state='normal')
        else:
            self.drag_predict_btn.config(state='disabled')

    def predict_from_drag_drop(self):
        """Predict archetype from drag-drop deck"""
        if not self.predictor:
            messagebox.showerror("Error", "Model not loaded. Please train a model first.")
            return

        card_names = self.drag_drop_gui.get_deck_card_names()

        if len(card_names) != 8:
            messagebox.showwarning("Incomplete Deck", "Please add exactly 8 cards to your deck.")
            return

        # Store the current deck for detailed display
        card_ids = []
        for name in card_names:
            card_id = find_card_id_by_name(name)
            if card_id:
                card_ids.append(card_id)
        self.current_deck_card_ids = card_ids

        # Switch to results tab
        self.notebook.select(2)

        # Show loading
        self.show_loading_message()

        # Run prediction in background
        threading.Thread(target=self._predict_from_names_thread,
                         args=(card_names,), daemon=True).start()

    def clear_text_fields(self):
        """Clear all text input fields"""
        for entry in self.card_entries:
            entry.delete(0, tk.END)

        self.url_entry.delete(0, tk.END)
        self.ids_entry.delete(0, tk.END)

    def on_method_change(self):
        """Show/hide input sections based on selected method"""
        method = self.input_method.get()

        # Hide all frames first
        self.names_frame.pack_forget()
        self.url_frame.pack_forget()
        self.ids_frame.pack_forget()

        # Show selected frame
        if method == "names":
            self.names_frame.pack(fill=tk.X, pady=(10, 0))
        elif method == "url":
            self.url_frame.pack(fill=tk.X, pady=(10, 0))
        elif method == "ids":
            self.ids_frame.pack(fill=tk.X, pady=(10, 0))

    def fill_example_deck(self):
        """Fill with an example deck"""
        example_cards = [
            "MegaKnight", "PEKKA", "Bandit", "Royal Ghost",
            "eWiz", "Zap", "Poison", "Tornado"
        ]

        for i, card_name in enumerate(example_cards):
            if i < len(self.card_entries):
                self.card_entries[i].delete(0, tk.END)
                self.card_entries[i].insert(0, card_name)

    def load_model_background(self):
        """Load model in background thread"""

        def load_model():
            try:
                self.predictor = QuickClashPredictor("clash_royale_classifier.pth")
                self.root.after(0, self.on_model_loaded, True, "Model loaded successfully!")
            except FileNotFoundError:
                self.root.after(0, self.on_model_loaded, False, "Model file not found. Train a model first.")
            except Exception as e:
                self.root.after(0, self.on_model_loaded, False, f"Error loading model: {str(e)}")

        threading.Thread(target=load_model, daemon=True).start()

    def on_model_loaded(self, success, message):
        """Callback when model loading completes"""
        self.load_model_attempted = True

        if success:
            self.status_label.config(text=message)
            self.predict_btn.config(state='normal')
            self.drag_predict_btn.config(state='normal')
        else:
            self.status_label.config(text=message)
            self.predict_btn.config(state='disabled')
            self.drag_predict_btn.config(state='disabled')

    def predict_archetype(self):
        """Predict archetype based on input method"""
        if not self.predictor:
            messagebox.showerror("Error", "Model not loaded. Please train a model first.")
            return

        method = self.input_method.get()

        try:
            if method == "names":
                self.predict_from_names()
            elif method == "url":
                self.predict_from_url()
            elif method == "ids":
                self.predict_from_ids()
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction: {str(e)}")

    def predict_from_names(self):
        """Predict from card names (remove duplicates from text inputs)."""
        # Collect non-empty names preserving order
        raw_names = [entry.get().strip() for entry in self.card_entries if entry.get().strip()]
        # Deduplicate case-insensitively, keep first occurrence
        seen = set()
        unique_names = []
        for n in raw_names:
            key = n.lower()
            if key not in seen:
                seen.add(key)
                unique_names.append(n)

        # If duplicates were removed, update the text fields to reflect cleaned list
        if len(unique_names) != len(raw_names):
            for i, entry in enumerate(self.card_entries):
                entry.delete(0, tk.END)
                if i < len(unique_names):
                    entry.insert(0, unique_names[i])
            messagebox.showinfo("Duplicates Removed", "Duplicate card names were removed from the input fields.")

        if len(unique_names) != 8:
            messagebox.showwarning("Input Error", "Please enter exactly 8 unique card names.")
            return

        # Switch to results tab and run prediction
        self.notebook.select(2)
        self.show_loading_message()
        threading.Thread(target=self._predict_from_names_thread,
                         args=(unique_names,), daemon=True).start()

    def _predict_from_names_thread(self, card_names):
        """Background thread for name-based prediction"""
        try:
            # Store the current deck for detailed display
            card_ids = []
            for name in card_names:
                card_id = find_card_id_by_name(name)
                if card_id:
                    card_ids.append(card_id)
            self.current_deck_card_ids = card_ids

            result = self.predictor.predict_from_card_names(card_names)
            self.root.after(0, self.display_prediction_result, result)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def predict_from_url(self):
        """Predict from deck URL"""
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Input Error", "Please enter a deck URL.")
            return

        # Switch to results tab
        self.notebook.select(2)

        # Show loading
        self.show_loading_message()

        # Run prediction in background
        threading.Thread(target=self._predict_from_url_thread,
                         args=(url,), daemon=True).start()




    def predict_from_ids(self):
        """Predict from card IDs (semicolon-separated) with duplicate removal."""
        ids_text = self.ids_entry.get().strip()
        if not ids_text:
            messagebox.showwarning("Input Error", "Please enter card IDs.")
            return

        parts = [p.strip() for p in ids_text.split(';') if p.strip()]
        seen = set()
        unique_ids = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                unique_ids.append(p)

        # If duplicates were removed, update the IDs entry
        if len(unique_ids) != len(parts):
            cleaned = ';'.join(unique_ids)
            self.ids_entry.delete(0, tk.END)
            self.ids_entry.insert(0, cleaned)
            messagebox.showinfo("Duplicates Removed", "Duplicate card IDs were removed from the input field.")

        if len(unique_ids) != 8:
            messagebox.showwarning("Input Error", "Please enter exactly 8 unique card IDs.")
            return

        # Switch to results tab and run prediction
        self.notebook.select(2)
        self.show_loading_message()
        threading.Thread(target=self._predict_from_ids_thread,
                         args=(';'.join(unique_ids),), daemon=True).start()

    def _predict_from_url_thread(self, url):
        """Background thread for URL-based prediction with duplicate removal."""
        try:
            # Extract deck ids and dedupe while preserving order
            deck = self.predictor.trainer.processor.extract_deck_from_url(url)
            seen = set()
            deduped_deck = []
            for cid in deck:
                if cid not in seen:
                    seen.add(cid)
                    deduped_deck.append(cid)

            if len(deduped_deck) != len(deck):
                # Inform user on main thread that duplicates were removed
                self.root.after(0, lambda: messagebox.showinfo("Duplicates Removed",
                                                               "Duplicate card IDs found in URL were removed before prediction."))

            if len(deduped_deck) != 8:
                self.root.after(0, lambda: messagebox.showwarning("Input Error",
                                                                  "Deck must contain exactly 8 unique cards after removing duplicates."))
                return

            # Store the current deck and run prediction using deck string
            self.current_deck_card_ids = deduped_deck
            deck_str = ';'.join(map(str, deduped_deck))
            result = self.predictor.predict_from_deck_string(deck_str)
            self.root.after(0, self.display_prediction_result, result)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def _predict_from_ids_thread(self, ids_text):
        """Background thread for ID-based prediction"""
        try:
            # Store the current deck for detailed display
            deck = self.predictor.trainer.processor.extract_from_deck_string(ids_text)
            self.current_deck_card_ids = deck

            result = self.predictor.predict_from_deck_string(ids_text)
            self.root.after(0, self.display_prediction_result, result)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def show_loading_message(self):
        """Show loading message in results area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Analyzing deck... Please wait...")
        self.results_text.config(state=tk.DISABLED)

    def display_prediction_result(self, result):
        """Display prediction results in the text area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        if 'error' in result:
            self.results_text.insert(tk.END, f"Error: {result['error']}")
        else:
            # Get deck stats and card details
            deck = []
            if hasattr(self, 'current_deck_card_ids'):
                deck = self.current_deck_card_ids
            elif 'deck_stats' in result:
                # Extract card IDs from deck stats if available
                deck = [card['id'] for card in result['deck_stats']['card_details']]

            # Calculate deck stats if not already in result
            if 'deck_stats' not in result and deck:
                result['deck_stats'] = calculate_deck_stats(deck)

            # Format the results with all the detailed information
            output = f"Model loaded with {len(self.predictor.trainer.processor.card_id_to_index)} unique cards and {len(self.predictor.trainer.processor.archetypes)} archetypes\n\n"
            output += "=" * 50 + "\n"
            output += "DECK ANALYSIS\n"
            output += "=" * 50 + "\n\n"

            output += f"Archetype: {result['archetype']}\n"
            output += f"Confidence: {result['confidence']:.2%}\n\n"

            # Deck statistics
            if 'deck_stats' in result:
                stats = result['deck_stats']
                output += f"Average Elixir Cost: {stats['average_elixir']:.2f}\n"
                output += f"4-Card Cycle Cost: {stats['four_card_cycle']}\n"
                output += f"Total Deck Cost: {stats['total_elixir']}\n\n"

            # Card type distribution
            card_types = result.get('card_types', {})
            output += f"Card Types: {card_types}\n\n"

            # Deck composition
            output += "Deck Composition:\n"
            output += "-" * 40 + "\n"

            if 'deck_stats' in result and 'card_details' in result['deck_stats']:
                rarity_names = {1: "Common", 2: "Rare", 3: "Epic", 4: "Legendary", 5: "Champion"}
                for i, card in enumerate(result['deck_stats']['card_details'], 1):
                    output += f"{i}. {card['name']} ({card['elixir']} elixir) - {card['type'].title()} - {rarity_names[card['rarity']]}\n"
            else:
                # Fallback: try to get card details from the deck
                if deck:
                    rarity_names = {1: "Common", 2: "Rare", 3: "Epic", 4: "Legendary", 5: "Champion"}
                    for i, card_id in enumerate(deck, 1):
                        card_info = get_card_info(card_id)
                        output += f"{i}. {card_info['name']} ({card_info['elixir']} elixir) - {card_info['type'].title()} - {rarity_names[card_info['rarity']]}\n"

            output += "\n"

            # All probabilities
            output += "All Archetype Probabilities:\n"
            output += "-" * 30 + "\n"
            all_probs = result.get('all_probabilities', {})
            for arch, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                output += f"  {arch}: {prob:.2%}\n"

            self.results_text.insert(tk.END, output)

            # Update the Deck overview tab chart and images
            try:
                self.update_deck_overview(all_probs, deck)
            except Exception:
                pass

        self.results_text.config(state=tk.DISABLED)

    def display_error(self, error_message):
        """Display error message in results area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"❌ ERROR:\n{error_message}")
        self.results_text.config(state=tk.DISABLED)

    def train_model(self):
        """Train a new model"""
        if not messagebox.askyesno("Confirm Training",
                                   "Training a new model may take several minutes. Continue?"):
            return

        # Disable predict button and show progress
        self.predict_btn.config(state='disabled')
        self.drag_predict_btn.config(state='disabled')
        self.train_btn.config(state='disabled')
        self.progress.start()
        self.status_label.config(text="Training model...")

        # Run training in background
        threading.Thread(target=self._train_model_thread, daemon=True).start()

    def _train_model_thread(self):
        """Background thread for model training"""
        try:
            train_new_model()
            self.root.after(0, self.on_training_completed, True, "Training completed successfully!")
        except Exception as e:
            self.root.after(0, self.on_training_completed, False, f"Training failed: {str(e)}")

    def on_training_completed(self, success, message):
        """Callback when training completes"""
        self.progress.stop()
        self.train_btn.config(state='normal')
        self.status_label.config(text=message)

        if success:
            # Reload the model
            self.load_model_background()
            messagebox.showinfo("Training Complete", message)
        else:
            messagebox.showerror("Training Failed", message)


def main():
    """Main function to run the GUI"""
    try:
        root = tk.Tk()
        app = ClashRoyaleGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {e}")


if __name__ == "__main__":
    main()