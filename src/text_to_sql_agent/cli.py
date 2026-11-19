"""Command-line interface: `text2sql "votre question"`."""
from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from .agent import TextToSQLAgent
from .config import PROJECT_ROOT, load_config

console = Console()


def _print_answer(question: str, agent: TextToSQLAgent, explain: bool) -> None:
    answer = agent.ask(question, explain=explain)

    console.print(f"\n[bold cyan]Question :[/] {answer.question}")
    console.print(f"[bold cyan]Ranker   :[/] {answer.ranker_name}\n")

    sql_panel = f"[yellow]{answer.sql}[/yellow]"
    console.print("[bold]SQL exécuté :[/]")
    console.print(sql_panel)

    table = Table(title=f"Résultat ({len(answer.df)} lignes)", show_lines=False)
    for col in answer.df.columns:
        table.add_column(str(col), style="green")
    for _, row in answer.df.iterrows():
        table.add_row(*[str(v) for v in row.values])
    console.print(table)

    if explain and answer.candidates:
        ct = Table(title="Candidats classés par le ranker", show_lines=False)
        ct.add_column("#")
        ct.add_column("score")
        ct.add_column("origin")
        ct.add_column("intent")
        ct.add_column("valide")
        ct.add_column("lignes")
        ct.add_column("SQL")
        for i, c in enumerate(answer.candidates, 1):
            ct.add_row(str(i), f"{c.score:.3f}", c.origin, c.intent, str(c.valid), str(c.n_rows), c.sql[:80])
        console.print(ct)


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(PROJECT_ROOT))  # rend `scripts.*` importable depuis n'importe où
    parser = argparse.ArgumentParser(prog="text2sql", description="Text-to-SQL agent avec ranking appris")
    parser.add_argument("question", nargs="*", help="Question en langage naturel (fr ou en)")
    parser.add_argument("--explain", action="store_true", help="Afficher le classement complet des candidats")
    parser.add_argument("--json", action="store_true", help="Sortie JSON")
    parser.add_argument("--build-data", action="store_true", help="(Re)construire la base 81k lignes puis quitter")
    parser.add_argument("--generate-train-data", action="store_true", help="Générer les données d'entraînement du ranker puis quitter")
    parser.add_argument("--train-ranker", action="store_true", help="Entraîner + évaluer le ranker puis quitter")
    parser.add_argument("--config", default=None, help="Chemin vers un config.yaml alternatif")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.build_data:
        from .database import build_database
        counts = build_database(str(cfg.resolve("database_path")), cfg.database_seed)
        console.print(f"[green]Base construite :[/] {counts} (total {sum(counts.values()):,} lignes)")
        return 0

    if args.generate_train_data:
        import scripts.generate_training_data as g  # noqa: PLC0415
        g.main(cfg)
        return 0

    if args.train_ranker:
        import scripts.train_ranker as t  # noqa: PLC0415
        return t.main(cfg)

    if not args.question:
        parser.print_help()
        return 1

    question = " ".join(args.question)
    agent = TextToSQLAgent(cfg)
    try:
        if args.json:
            answer = agent.ask(question, explain=True)
            payload = {
                "question": answer.question,
                "sql": answer.sql,
                "ranker": answer.ranker_name,
                "result": answer.df.to_dict(orient="records"),
                "candidates": [
                    {"sql": c.sql, "score": c.score, "valid": c.valid, "origin": c.origin, "intent": c.intent}
                    for c in answer.candidates
                ],
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            _print_answer(question, agent, explain=args.explain)
        return 0
    except ValueError as exc:
        console.print(f"[red]Erreur :[/] {exc}")
        return 1
    finally:
        agent.close()


if __name__ == "__main__":
    sys.exit(main())
