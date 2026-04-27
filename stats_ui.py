from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from config import config
from event_store import (
    CATEGORY_ACE,
    CATEGORY_BALL_ON_NET,
    CATEGORY_BLOCK,
    CATEGORY_ERROR,
    CATEGORY_FREEBALL,
    CATEGORY_SPIKE,
    CATEGORY_UNDEFINED,
    EventStore,
)


APP_BG = "#F5F7FA"
SIDEBAR_BG = "#1F2937"
SIDEBAR_BTN = "#111827"
SIDEBAR_HOVER = "#374151"
PRIMARY = "#2563EB"
PRIMARY_SOFT = "#EFF6FF"
CARD_BG = "#FFFFFF"
CARD_ALT = "#F9FAFB"
BORDER = "#E5E7EB"
BORDER_STRONG = "#BFDBFE"
TEXT = "#111827"
TEXT_SOFT = "#6B7280"
TEXT_FAINT = "#9CA3AF"
TEAM_A_BG = "#DBEAFE"
TEAM_A_FG = "#1D4ED8"
TEAM_B_BG = "#FCE7F3"
TEAM_B_FG = "#BE185D"
NEUTRAL_BG = "#E5E7EB"
NEUTRAL_FG = "#374151"
FONT = "Segoe UI"
FONT_EMOJI = "Segoe UI Emoji"

META = {
    CATEGORY_SPIKE: {"label": "Spikes", "singular": "Spike", "icon": "\U0001F525", "badge_bg": "#FFEDD5", "badge_fg": "#C2410C"},
    CATEGORY_BLOCK: {"label": "Blocos", "singular": "Bloco", "icon": "\U0001F9F1", "badge_bg": "#DBEAFE", "badge_fg": "#1D4ED8"},
    CATEGORY_ACE: {"label": "Aces", "singular": "Ace", "icon": "\U0001F3AF", "badge_bg": "#FEF3C7", "badge_fg": "#92400E"},
    CATEGORY_ERROR: {"label": "Erros", "singular": "Erro", "icon": "\u26a0", "badge_bg": "#FEE2E2", "badge_fg": "#B91C1C"},
    CATEGORY_FREEBALL: {"label": "Freeballs", "singular": "Freeball", "icon": "\U0001F4A8", "badge_bg": "#D1FAE5", "badge_fg": "#047857"},
    CATEGORY_BALL_ON_NET: {"label": "Bola na Rede", "singular": "Bola na Rede", "icon": "\U0001F3D0", "badge_bg": "#E0E7FF", "badge_fg": "#3730A3"},
    CATEGORY_UNDEFINED: {"label": "Indefinidos", "singular": "Indefinido", "icon": "\u2753", "badge_bg": "#EDE9FE", "badge_fg": "#6D28D9"},
}
ORDER = (
    CATEGORY_SPIKE,
    CATEGORY_BLOCK,
    CATEGORY_ACE,
    CATEGORY_ERROR,
    CATEGORY_FREEBALL,
    CATEGORY_BALL_ON_NET,
    CATEGORY_UNDEFINED,
)


class StatsUI(tk.Tk):
    def __init__(self, store_path: Path):
        super().__init__()
        self.store_path = Path(store_path)
        self.title("Voleibol - Dashboard de Eventos")
        self.geometry("1450x860")
        self.minsize(1180, 720)
        self.configure(bg=APP_BG)

        self.current_category = CATEGORY_SPIKE
        self.current_events = []
        self.selected_event = None
        self.selected_event_id = None
        self.preview_image = None
        self.category_widgets = {}
        self.event_widgets = {}
        self.hover_category = None
        self.hover_event = None
        self.store_api = EventStore(
            store_path=self.store_path,
            preview_dir=Path(config.event_preview_dir),
            source_video_path=None,
            reset_on_start=False,
            preview_max_width=int(getattr(config, "event_preview_max_width", 420)),
        )

        self.status_var = tk.StringVar(value="0 eventos")
        self.updated_var = tk.StringVar(value="Sem atualizacoes")
        self.list_title_var = tk.StringVar(value="Spikes")
        self.list_subtitle_var = tk.StringVar(value="Selecione um evento para ver os detalhes.")

        self._styles()
        self._layout()
        self.refresh_data()
        self.after(3000, self._auto_refresh)

    def _styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Modern.Vertical.TScrollbar",
            troughcolor=APP_BG,
            background="#CBD5E1",
            bordercolor=APP_BG,
            arrowcolor=TEXT_SOFT,
            darkcolor="#CBD5E1",
            lightcolor="#CBD5E1",
            relief="flat",
            gripcount=0,
        )

    def _layout(self) -> None:
        self.grid_columnconfigure(0, weight=0, minsize=260)
        self.grid_columnconfigure(1, weight=5)
        self.grid_columnconfigure(2, weight=6)
        self.grid_rowconfigure(1, weight=1)
        self._header()
        self._sidebar()
        self._event_panel()
        self._detail_panel()

    def _card(self, parent, bg=CARD_BG, border=BORDER) -> tk.Frame:
        return tk.Frame(parent, bg=bg, highlightbackground=border, highlightthickness=1)

    def _badge(self, parent, text, bg, fg, font=(FONT, 9), padx=10, pady=4):
        return tk.Label(parent, text=text, bg=bg, fg=fg, font=font, padx=padx, pady=pady)

    def _button(self, parent, text, command, primary=False) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=(FONT, 10, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            bg=PRIMARY if primary else CARD_ALT,
            fg="#FFFFFF" if primary else TEXT,
            activebackground="#1D4ED8" if primary else PRIMARY_SOFT,
            activeforeground="#FFFFFF" if primary else PRIMARY,
            padx=16,
            pady=9,
        )

    def _header(self) -> None:
        row = tk.Frame(self, bg=APP_BG)
        row.grid(row=0, column=0, columnspan=3, sticky="ew", padx=24, pady=(24, 18))
        row.grid_columnconfigure(0, weight=1)
        card = self._card(row)
        card.pack(fill="x")
        left = tk.Frame(card, bg=CARD_BG)
        left.pack(side="left", fill="x", expand=True, padx=24, pady=20)
        tk.Label(left, text="Dashboard de Eventos", font=(FONT, 20, "bold"), fg=TEXT, bg=CARD_BG).pack(anchor="w")
        tk.Label(
            left,
            text="Interface de consulta para spikes, blocos e eventos indefinidos.",
            font=(FONT, 10),
            fg=TEXT_SOFT,
            bg=CARD_BG,
        ).pack(anchor="w", pady=(6, 0))
        right = tk.Frame(card, bg=CARD_BG)
        right.pack(side="right", padx=24, pady=18)
        tk.Label(right, textvariable=self.status_var, font=(FONT, 9), fg=PRIMARY, bg=PRIMARY_SOFT, padx=12, pady=8).pack(side="left")
        tk.Label(right, textvariable=self.updated_var, font=(FONT, 9), fg=TEXT_SOFT, bg=CARD_ALT, padx=12, pady=8).pack(side="left", padx=(10, 12))
        self._button(right, "Atualizar", self.refresh_data, primary=True).pack(side="left")

    def _sidebar(self) -> None:
        side = tk.Frame(self, bg=SIDEBAR_BG, width=260)
        side.grid(row=1, column=0, sticky="ns", padx=(24, 18), pady=(0, 24))
        side.grid_propagate(False)
        tk.Label(side, text="Categorias", font=(FONT, 13, "bold"), fg="#F9FAFB", bg=SIDEBAR_BG).pack(anchor="w", padx=22, pady=(24, 4))
        tk.Label(side, text="Filtra os eventos guardados.", font=(FONT, 9), fg="#9CA3AF", bg=SIDEBAR_BG).pack(anchor="w", padx=22, pady=(0, 18))
        box = tk.Frame(side, bg=SIDEBAR_BG)
        box.pack(fill="x", padx=18)
        for category in ORDER:
            meta = META[category]
            wrapper = tk.Frame(box, bg=SIDEBAR_BG)
            wrapper.pack(fill="x", pady=(0, 10))
            btn = tk.Frame(wrapper, bg=SIDEBAR_BTN, cursor="hand2", padx=16, pady=14)
            btn.pack(fill="x")
            btn.grid_columnconfigure(1, weight=1)
            icon = tk.Label(btn, text=meta["icon"], font=(FONT_EMOJI, 16), fg="#FFFFFF", bg=SIDEBAR_BTN)
            icon.grid(row=0, column=0, rowspan=2, sticky="w")
            title = tk.Label(btn, text=meta["label"], font=(FONT, 10, "bold"), fg="#F9FAFB", bg=SIDEBAR_BTN)
            title.grid(row=0, column=1, sticky="w", padx=(12, 8))
            sub = tk.Label(btn, text=f"Eventos {meta['singular'].lower()}", font=(FONT, 8), fg="#9CA3AF", bg=SIDEBAR_BTN)
            sub.grid(row=1, column=1, sticky="w", padx=(12, 8), pady=(2, 0))
            count = tk.Label(btn, text="0", font=(FONT, 9), fg="#F9FAFB", bg="#374151", padx=10, pady=4)
            count.grid(row=0, column=2, rowspan=2, sticky="e")
            self.category_widgets[category] = {"btn": btn, "icon": icon, "title": title, "sub": sub, "count": count}
            for widget in (btn, icon, title, sub, count):
                widget.bind("<Button-1>", lambda _e, c=category: self.set_category(c))
                widget.bind("<Enter>", lambda _e, c=category: self._set_hover_category(c))
                widget.bind("<Leave>", lambda _e: self._set_hover_category(None))
        tk.Label(side, text="Fonte dos dados", font=(FONT, 8), fg="#9CA3AF", bg=SIDEBAR_BG).pack(anchor="w", padx=22, pady=(18, 2))
        tk.Label(side, text=str(self.store_path), font=(FONT, 8), fg="#D1D5DB", bg=SIDEBAR_BG, wraplength=210, justify="left").pack(anchor="w", padx=22)

    def _event_panel(self) -> None:
        panel = tk.Frame(self, bg=APP_BG)
        panel.grid(row=1, column=1, sticky="nsew", pady=(0, 24))
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)
        head = self._card(panel)
        head.grid(row=0, column=0, sticky="ew", padx=(0, 18), pady=(0, 16))
        tk.Label(head, textvariable=self.list_title_var, font=(FONT, 13, "bold"), fg=TEXT, bg=CARD_BG).pack(anchor="w", padx=20, pady=(18, 4))
        tk.Label(head, textvariable=self.list_subtitle_var, font=(FONT, 9), fg=TEXT_SOFT, bg=CARD_BG).pack(anchor="w", padx=20, pady=(0, 18))
        wrap = tk.Frame(panel, bg=APP_BG)
        wrap.grid(row=1, column=0, sticky="nsew", padx=(0, 18))
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)
        self.event_canvas = tk.Canvas(wrap, bg=APP_BG, highlightthickness=0, bd=0)
        self.event_canvas.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(wrap, orient="vertical", command=self.event_canvas.yview, style="Modern.Vertical.TScrollbar")
        bar.grid(row=0, column=1, sticky="ns")
        self.event_canvas.configure(yscrollcommand=bar.set)
        self.event_body = tk.Frame(self.event_canvas, bg=APP_BG)
        self.event_window = self.event_canvas.create_window((0, 0), window=self.event_body, anchor="nw")
        self.event_body.bind("<Configure>", lambda _e: self.event_canvas.configure(scrollregion=self.event_canvas.bbox("all")))
        self.event_canvas.bind("<Configure>", lambda e: self.event_canvas.itemconfigure(self.event_window, width=e.width))

    def _detail_panel(self) -> None:
        wrap = tk.Frame(self, bg=APP_BG)
        wrap.grid(row=1, column=2, sticky="nsew", padx=(0, 24), pady=(0, 24))
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)
        self.detail_canvas = tk.Canvas(wrap, bg=APP_BG, highlightthickness=0, bd=0)
        self.detail_canvas.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(wrap, orient="vertical", command=self.detail_canvas.yview, style="Modern.Vertical.TScrollbar")
        bar.grid(row=0, column=1, sticky="ns")
        self.detail_canvas.configure(yscrollcommand=bar.set)
        self.detail_body = tk.Frame(self.detail_canvas, bg=APP_BG)
        self.detail_window = self.detail_canvas.create_window((0, 0), window=self.detail_body, anchor="nw")
        self.detail_body.bind("<Configure>", lambda _e: self.detail_canvas.configure(scrollregion=self.detail_canvas.bbox("all")))
        self.detail_canvas.bind("<Configure>", lambda e: self.detail_canvas.itemconfigure(self.detail_window, width=e.width))

    def refresh_data(self) -> None:
        keep_id = self.selected_event_id
        snapshot = EventStore.load_snapshot(self.store_path)
        self.snapshot = snapshot
        self.store_api.data = snapshot
        events = snapshot.get("events", [])
        self.status_var.set(f"{len(events)} eventos")
        self.updated_var.set(f"Atualizado: {snapshot.get('updated_at') or '--'}")
        counts = {cat: 0 for cat in ORDER}
        for event in events:
            counts[str(event.get("category", CATEGORY_UNDEFINED))] = counts.get(str(event.get("category", CATEGORY_UNDEFINED)), 0) + 1
        for category in ORDER:
            self.category_widgets[category]["count"].configure(text=str(counts.get(category, 0)))
        self.set_category(self.current_category, keep_id)

    def set_category(self, category: str, preserve_event_id=None) -> None:
        self.current_category = category
        self.list_title_var.set(f"{META[category]['icon']}  {META[category]['label']}")
        self.current_events = [event for event in getattr(self, "snapshot", {}).get("events", []) if str(event.get("category")) == category]
        self._paint_sidebar()
        self._render_event_cards()
        next_id = None
        if preserve_event_id is not None:
            for event in self.current_events:
                if int(event.get("id", -1)) == int(preserve_event_id):
                    next_id = int(preserve_event_id)
                    break
        if next_id is None and self.current_events:
            next_id = int(self.current_events[0].get("id"))
        self.select_event(next_id)

    def _render_event_cards(self) -> None:
        for child in self.event_body.winfo_children():
            child.destroy()
        self.event_widgets.clear()
        self.list_subtitle_var.set("Lista cronológica dos eventos guardados nesta categoria.")
        if not self.current_events:
            card = self._card(self.event_body)
            card.pack(fill="x", pady=(0, 12))
            tk.Label(card, text="Sem eventos nesta categoria", font=(FONT, 12, "bold"), fg=TEXT, bg=CARD_BG).pack(anchor="w", padx=20, pady=(18, 6))
            tk.Label(card, text="Novos eventos aparecem aqui automaticamente quando o pipeline os guardar.", font=(FONT, 9), fg=TEXT_SOFT, bg=CARD_BG, wraplength=420, justify="left").pack(anchor="w", padx=20, pady=(0, 18))
            return
        for event in self.current_events:
            event_id = int(event.get("id", 0))
            meta = META.get(str(event.get("category")), META[CATEGORY_UNDEFINED])
            shell = tk.Frame(self.event_body, bg=APP_BG)
            shell.pack(fill="x", pady=(0, 12))
            card = tk.Frame(shell, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1, cursor="hand2")
            card.pack(fill="x")
            stripe = tk.Frame(card, bg=CARD_BG, width=8)
            stripe.pack(side="left", fill="y")
            body = tk.Frame(card, bg=CARD_BG)
            body.pack(side="left", fill="both", expand=True, padx=(14, 16), pady=14)
            top = tk.Frame(body, bg=CARD_BG)
            top.pack(fill="x")
            tk.Label(top, text=self._event_title(event), font=(FONT, 12, "bold"), fg=TEXT, bg=CARD_BG).pack(side="left")
            conf = self._badge(top, f"Conf. {self._fmt_conf(event.get('confidence'))}", meta["badge_bg"], meta["badge_fg"], font=(FONT, 8))
            conf.pack(side="right")
            mid = tk.Frame(body, bg=CARD_BG)
            mid.pack(fill="x", pady=(10, 0))
            self._badge(mid, self._team(event.get("point_team")), self._team_bg(event.get("point_team")), self._team_fg(event.get("point_team")), font=(FONT, 8)).pack(side="left")
            tk.Label(mid, text=event.get("timestamp_label") or "--:--:--", font=(FONT, 9), fg=TEXT_SOFT, bg=CARD_BG).pack(side="right")
            tk.Label(body, text=self._meta_line(event), font=(FONT, 9), fg=TEXT_SOFT, bg=CARD_BG).pack(anchor="w", pady=(12, 0))
            self.event_widgets[event_id] = {"card": card, "stripe": stripe, "body": body}
            for widget in (card, body, top, mid):
                widget.bind("<Button-1>", lambda _e, eid=event_id: self.select_event(eid))
                widget.bind("<Enter>", lambda _e, eid=event_id: self._set_hover_event(eid))
                widget.bind("<Leave>", lambda _e: self._set_hover_event(None))
            for widget in top.winfo_children() + mid.winfo_children():
                widget.bind("<Button-1>", lambda _e, eid=event_id: self.select_event(eid))
                widget.bind("<Enter>", lambda _e, eid=event_id: self._set_hover_event(eid))
                widget.bind("<Leave>", lambda _e: self._set_hover_event(None))
        self._paint_events()

    def select_event(self, event_id) -> None:
        self.selected_event_id = None if event_id is None else int(event_id)
        self.selected_event = None if self.selected_event_id is None else self.store_api.get_event(self.selected_event_id)
        self._paint_events()
        self._render_details(self.selected_event)

    def _paint_sidebar(self) -> None:
        for category, widgets in self.category_widgets.items():
            active = category == self.current_category
            hover = category == self.hover_category
            bg = PRIMARY if active else SIDEBAR_HOVER if hover else SIDEBAR_BTN
            sub = "#DBEAFE" if active else "#D1D5DB" if hover else "#9CA3AF"
            pill = "#1D4ED8" if active else "#4B5563" if hover else "#374151"
            for key in ("btn", "icon", "title", "sub"):
                widgets[key].configure(bg=bg)
            widgets["sub"].configure(fg=sub)
            widgets["count"].configure(bg=pill, fg="#F9FAFB")

    def _paint_events(self) -> None:
        for event_id, widgets in self.event_widgets.items():
            selected = event_id == self.selected_event_id
            hover = event_id == self.hover_event
            bg = PRIMARY_SOFT if selected else CARD_ALT if hover else CARD_BG
            border = BORDER_STRONG if selected else BORDER
            widgets["card"].configure(bg=bg, highlightbackground=border)
            widgets["stripe"].configure(bg=PRIMARY if selected else bg)
            widgets["body"].configure(bg=bg)
            for child in widgets["body"].winfo_children():
                child.configure(bg=bg)
                for nested in child.winfo_children():
                    if isinstance(nested, tk.Label) and nested.cget("bg") not in (TEAM_A_BG, TEAM_B_BG, NEUTRAL_BG, META[CATEGORY_SPIKE]["badge_bg"], META[CATEGORY_BLOCK]["badge_bg"], META[CATEGORY_UNDEFINED]["badge_bg"]):
                        nested.configure(bg=bg)

    def _render_details(self, event: dict | None) -> None:
        for child in self.detail_body.winfo_children():
            child.destroy()
        self.preview_image = None
        if event is None:
            card = self._card(self.detail_body)
            card.pack(fill="x", pady=(0, 12))
            tk.Label(card, text="Nenhum evento selecionado", font=(FONT, 13, "bold"), fg=TEXT, bg=CARD_BG).pack(anchor="w", padx=22, pady=(22, 8))
            tk.Label(card, text="Escolhe um evento na lista do centro para ver o detalhe e o preview.", font=(FONT, 9), fg=TEXT_SOFT, bg=CARD_BG).pack(anchor="w", padx=22, pady=(0, 22))
            return
        meta = META.get(str(event.get("category")), META[CATEGORY_UNDEFINED])
        hero = self._card(self.detail_body)
        hero.pack(fill="x", pady=(0, 16))
        top = tk.Frame(hero, bg=CARD_BG)
        top.pack(fill="x", padx=22, pady=(20, 8))
        tk.Label(top, text=self._event_title(event), font=(FONT, 18, "bold"), fg=TEXT, bg=CARD_BG).pack(side="left")
        self._badge(top, f"{meta['icon']} {meta['singular']}", meta["badge_bg"], meta["badge_fg"], font=(FONT, 9), padx=12, pady=6).pack(side="right")
        tk.Label(hero, text=f"Timestamp {event.get('timestamp_label') or '--:--:--'}  •  {self._point_type(event.get('point_type'))}", font=(FONT, 9), fg=TEXT_SOFT, bg=CARD_BG).pack(anchor="w", padx=22, pady=(0, 14))
        chips = tk.Frame(hero, bg=CARD_BG)
        chips.pack(fill="x", padx=22, pady=(0, 22))
        self._badge(chips, self._team(event.get("point_team")), self._team_bg(event.get("point_team")), self._team_fg(event.get("point_team")), font=(FONT, 9), padx=12, pady=6).pack(side="left", padx=(0, 8))
        self._badge(chips, f"Confianca {self._fmt_conf(event.get('confidence'))}", "#DCFCE7" if self._conf_value(event.get("confidence")) >= 0.7 else "#FEF3C7", "#166534" if self._conf_value(event.get("confidence")) >= 0.7 else "#92400E", font=(FONT, 9), padx=12, pady=6).pack(side="left", padx=(0, 8))
        self._badge(chips, f"Score {self._score_line(event)}", NEUTRAL_BG, NEUTRAL_FG, font=(FONT, 9), padx=12, pady=6).pack(side="left")

        preview = self._card(self.detail_body)
        preview.pack(fill="x", pady=(0, 16))
        head = tk.Frame(preview, bg=CARD_BG)
        head.pack(fill="x", padx=22, pady=(18, 14))
        left = tk.Frame(head, bg=CARD_BG)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text="Preview do Evento", font=(FONT, 13, "bold"), fg=TEXT, bg=CARD_BG).pack(anchor="w")
        tk.Label(left, text="Frame representativo guardado pelo pipeline.", font=(FONT, 9), fg=TEXT_SOFT, bg=CARD_BG).pack(anchor="w", pady=(4, 0))
        actions = tk.Frame(head, bg=CARD_BG)
        actions.pack(side="right")
        self.open_preview_button = self._button(actions, "Abrir preview", self.open_preview, primary=False)
        self.open_preview_button.pack(side="left", padx=(0, 8))
        self.open_video_button = self._button(actions, "Ver video", self.open_video, primary=True)
        self.open_video_button.pack(side="left")
        shell = tk.Frame(preview, bg=CARD_BG)
        shell.pack(fill="x", padx=22, pady=(0, 22))
        box = tk.Frame(shell, bg=CARD_ALT, height=300, highlightbackground=BORDER, highlightthickness=1)
        box.pack(fill="x")
        box.pack_propagate(False)
        self.preview_label = tk.Label(box, text="Sem frame representativo", font=(FONT, 9), fg=TEXT_SOFT, bg=CARD_ALT)
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")
        preview_path = event.get("representative_frame_path")
        if preview_path and Path(preview_path).exists():
            self._render_preview(Path(preview_path))
            self.open_preview_button.configure(state="normal")
        else:
            self.open_preview_button.configure(state="disabled")
        video_path = event.get("source_video_path")
        self.open_video_button.configure(state="normal" if video_path and Path(video_path).exists() else "disabled")

        grid = tk.Frame(self.detail_body, bg=APP_BG)
        grid.pack(fill="x")
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)
        self._section(grid, "Informações Gerais", [("Tipo", META.get(str(event.get("category")), META[CATEGORY_UNDEFINED])["singular"]), ("Equipa", self._team(event.get("point_team"))), ("Confianca", self._fmt_conf(event.get("confidence"))), ("Frames", f"{event.get('start_frame', '--')} -> {event.get('end_frame', '--')}")]).grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 16))
        self._section(grid, "Tempo", [("Inicio", self._seconds(event.get("start_time_seconds"))), ("Fim", self._seconds(event.get("end_time_seconds"))), ("Duracao", self._seconds(event.get("rally_duration_seconds"))), ("Timestamp", event.get("timestamp_label") or "--:--:--")]).grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 16))
        self._section(grid, "Bola", [("Velocidade media", self._speed(event.get("ball_avg_speed"))), ("Velocidade maxima", self._speed(event.get("ball_max_speed"))), ("Point type", self._point_type(event.get("point_type")))]).grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 16))
        self._section(grid, "Jogo", [("Lado ataque", self._side(event.get("attack_side"))), ("Lado defesa", self._side(event.get("defending_side"))), ("Resultado", self._score_line(event))]).grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(0, 16))
        note = self._card(self.detail_body)
        note.pack(fill="x", pady=(0, 16))
        tk.Label(note, text="Classificação", font=(FONT, 13, "bold"), fg=TEXT, bg=CARD_BG).pack(anchor="w", padx=18, pady=(18, 4))
        tk.Label(note, text="Motivo e observações guardadas para este evento.", font=(FONT, 9), fg=TEXT_SOFT, bg=CARD_BG).pack(anchor="w", padx=18, pady=(0, 12))
        for label, value in (("Motivo", self._text(event.get("reason"))), ("Notas", self._text(event.get("notes"))), ("Video", self._text(event.get("source_video_path")))):
            block = tk.Frame(note, bg=CARD_BG)
            block.pack(fill="x", padx=18, pady=(0, 12))
            tk.Label(block, text=label, font=(FONT, 9), fg=TEXT_SOFT, bg=CARD_BG).pack(anchor="w")
            tk.Label(block, text=value, font=(FONT, 10), fg=TEXT, bg=CARD_BG, wraplength=560, justify="left").pack(anchor="w", pady=(4, 0))

    def _section(self, parent, title, rows) -> tk.Frame:
        shell = tk.Frame(parent, bg=APP_BG)
        card = self._card(shell)
        card.pack(fill="both", expand=True)
        tk.Label(card, text=title, font=(FONT, 13, "bold"), fg=TEXT, bg=CARD_BG).pack(anchor="w", padx=18, pady=(18, 12))
        body = tk.Frame(card, bg=CARD_BG)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        for i, (label, value) in enumerate(rows):
            row = tk.Frame(body, bg=CARD_BG)
            row.pack(fill="x", pady=(0, 10 if i < len(rows) - 1 else 0))
            tk.Label(row, text=label, font=(FONT, 9), fg=TEXT_SOFT, bg=CARD_BG).pack(side="left")
            tk.Label(row, text=value, font=(FONT, 10, "bold"), fg=TEXT, bg=CARD_BG).pack(side="right")
        return shell

    def _render_preview(self, path: Path) -> None:
        try:
            image = tk.PhotoImage(file=str(path))
        except tk.TclError:
            self.preview_label.configure(text="Nao foi possivel carregar o preview.", image="")
            self.preview_image = None
            return
        factor = max(1, int(math.ceil(max(image.width() / 520.0, image.height() / 280.0))))
        if factor > 1:
            image = image.subsample(factor, factor)
        self.preview_image = image
        self.preview_label.configure(image=image, text="")

    def _event_title(self, event: dict) -> str:
        return f"{META.get(str(event.get('category')), META[CATEGORY_UNDEFINED])['singular']} #{int(event.get('id', 0)):03d}"

    def _meta_line(self, event: dict) -> str:
        return f"{self._point_type(event.get('point_type'))}  •  ataque {self._side(event.get('attack_side'))}"

    @staticmethod
    def _team(value) -> str:
        return "Equipa A" if value == "TeamA" else "Equipa B" if value == "TeamB" else "--" if value in (None, "", "--") else str(value)

    @staticmethod
    def _side(value) -> str:
        return "Campo A" if value == "CampoA" else "Campo B" if value == "CampoB" else "--" if value in (None, "", "--") else str(value)

    @staticmethod
    def _team_bg(team) -> str:
        return TEAM_A_BG if team == "TeamA" else TEAM_B_BG if team == "TeamB" else NEUTRAL_BG

    @staticmethod
    def _team_fg(team) -> str:
        return TEAM_A_FG if team == "TeamA" else TEAM_B_FG if team == "TeamB" else NEUTRAL_FG

    @staticmethod
    def _point_type(value) -> str:
        mapping = {"POINT_BY_SPIKE": "Ponto por Spike", "POINT_BY_BLOCK": "Ponto por Bloco", "FREEBALL": "Freeball", "BOLA_NA_REDE": "Bola na Rede", "RALLY_ONLY": "Rally sem classificacao"}
        return "--" if value in (None, "", "--") else mapping.get(str(value), str(value).replace("_", " ").title())

    @staticmethod
    def _seconds(value) -> str:
        try:
            return "--" if value is None else f"{float(value):.2f}s"
        except (TypeError, ValueError):
            return "--"

    @staticmethod
    def _speed(value) -> str:
        try:
            return "--" if value is None else f"{float(value):.2f} px/frame"
        except (TypeError, ValueError):
            return "--"

    @staticmethod
    def _text(value) -> str:
        text = "" if value is None else str(value).strip()
        return text or "--"

    @staticmethod
    def _conf_value(value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _fmt_conf(self, value) -> str:
        conf = self._conf_value(value)
        return "--" if conf <= 0.0 else f"{conf * 100:.0f}%"

    def _score_line(self, event: dict) -> str:
        notes = str(event.get("notes") or "")
        for item in notes.split("|"):
            chunk = item.strip()
            if chunk.startswith("score:") and "->" in chunk:
                left, right = chunk[6:].split("->", 1)
                return f"{self._score_points(left)} -> {self._score_points(right)}"
        return "--"

    @staticmethod
    def _score_points(raw: str) -> str:
        try:
            parsed = ast.literal_eval(raw.strip())
            return f"{int(parsed[1])}-{int(parsed[3])}" if isinstance(parsed, (list, tuple)) and len(parsed) >= 4 else raw.strip()
        except (SyntaxError, ValueError, TypeError):
            return raw.strip() or "--"

    def _set_hover_category(self, category) -> None:
        self.hover_category = category
        self._paint_sidebar()

    def _set_hover_event(self, event_id) -> None:
        self.hover_event = event_id
        self._paint_events()

    def _auto_refresh(self) -> None:
        try:
            self.refresh_data()
        finally:
            self.after(3000, self._auto_refresh)

    def open_preview(self) -> None:
        if self.selected_event and self.selected_event.get("representative_frame_path"):
            try:
                os.startfile(self.selected_event["representative_frame_path"])  # type: ignore[attr-defined]
            except Exception:
                pass

    def open_video(self) -> None:
        if self.selected_event and self.selected_event.get("source_video_path"):
            try:
                os.startfile(self.selected_event["source_video_path"])  # type: ignore[attr-defined]
            except Exception:
                pass


def main():
    config.ensure_dirs()
    app = StatsUI(Path(config.event_store_file))
    app.mainloop()


if __name__ == "__main__":
    main()
