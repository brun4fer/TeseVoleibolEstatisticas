"""
stats_debug.py
--------------
Resumo rápido das estatísticas a partir de outputs/volleyball_events.json.

Mostra:
  - Contagem por categoria (spike, block, ace, error, freeball, ball_on_net,
    undefined) — globais e por equipa
  - Distribuição de point_type vs categoria (auditar mapeamento)
  - Top razões por categoria (para validar que as razões fazem sentido)
  - Lista de eventos recentes com campos críticos

Uso:
    python stats_debug.py                        # usa outputs/volleyball_events.json
    python stats_debug.py path/to/events.json    # caminho alternativo
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Dict, List


def load_events(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"[ERROR] Ficheiro não existe: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("events", [])


def section(title: str) -> None:
    # Encoding-safe (ASCII-only) para a consola Windows.
    safe = title.encode("ascii", errors="replace").decode("ascii")
    bar = "=" * len(safe)
    print(f"\n{safe}\n{bar}")


def per_team(events: List[Dict], category: str) -> Dict[str, int]:
    by = Counter()
    for e in events:
        if str(e.get("category")) != category:
            continue
        team = str(e.get("point_team") or "Desconhecido")
        by[team] += 1
    return dict(by)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/volleyball_events.json")
    events = load_events(path)

    section(f"Resumo @ {path}")
    print(f"Total de eventos: {len(events)}")

    if not events:
        print("(sem eventos registados)")
        return

    # Contagem por categoria
    section("Categorias (totais)")
    cats = Counter(str(e.get("category")) for e in events)
    for cat, count in cats.most_common():
        teams = per_team(events, cat)
        team_str = " | ".join(f"{t}:{c}" for t, c in sorted(teams.items())) if teams else "-"
        print(f"  {cat:12s} {count:3d}    [{team_str}]")

    # Distribuição point_type vs category — auditar mapeamento
    section("Mapping (category <- point_type)")
    cat_by_ptype = defaultdict(Counter)
    for e in events:
        cat_by_ptype[str(e.get("point_type"))][str(e.get("category"))] += 1
    for ptype, mapping in sorted(cat_by_ptype.items()):
        parts = " | ".join(f"{c}:{n}" for c, n in mapping.most_common())
        print(f"  {ptype:20s} -> {parts}")

    # Top reasons por categoria
    section("Top razões (por categoria)")
    reasons_by_cat = defaultdict(Counter)
    for e in events:
        reasons_by_cat[str(e.get("category"))][str(e.get("reason") or "-")] += 1
    for cat, reasons in sorted(reasons_by_cat.items()):
        print(f"  [{cat}]")
        for reason, count in reasons.most_common(5):
            print(f"      {count:3d}× {reason}")

    # Eventos recentes
    section("Últimos 10 eventos")
    for e in events[-10:]:
        cat = str(e.get("category", "?"))
        ptype = str(e.get("point_type", "?"))
        winner = str(e.get("point_team", "?"))
        ts = str(e.get("timestamp_label", "--:--:--"))
        conf = e.get("confidence")
        conf_s = f"{float(conf):.2f}" if conf is not None else "-"
        print(
            f"  #{int(e.get('id', 0)):03d}  {ts}  cat={cat:12s} ptype={ptype:20s} "
            f"team={winner:6s} conf={conf_s}"
        )

    # Cobertura — quantos pontos têm classificação útil
    section("Cobertura útil")
    useful = sum(1 for e in events if str(e.get("category")) != "undefined")
    pct = (useful / len(events)) * 100 if events else 0.0
    print(f"  {useful}/{len(events)} eventos classificados ({pct:.1f}%)")


if __name__ == "__main__":
    main()
