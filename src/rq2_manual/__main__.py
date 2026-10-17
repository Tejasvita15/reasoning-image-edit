"""Export RQ2 images for ChatGPT judging, import the judgments, and measure repeatability.

RQ2 counterpart of src/rq3_manual (same judge policy, prompt renderer, bundle
format and score validation). Workflow:

    1. export --kind calibration (20 fixed outputs, two passes)
    2. judge both passes; import each pass; repeatability -> MMD report
    3. export --kind main --calibration-report <report> (every output, one pass)
    4. judge; import pass 1
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("export", help="Make an offline ZIP with blinded images and prompts")
    command.add_argument("--logs", default="logs/rq2", help="Directory of RQ2 generation JSONL logs")
    command.add_argument("--splits", default="data/splits.json")
    command.add_argument("--split", default="k_dev", choices=("k_dev", "k_small", "k_test"))
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--official-checkout", default="external/Kris_Bench",
                         help="Clean checkout of mercurystraw/Kris_Bench at the pinned revision")
    command.add_argument("--output-dir", required=True)
    command.add_argument("--kind", choices=("calibration", "main"), default="calibration")
    command.add_argument("--limit", type=int, default=20, help="Calibration sample size")
    command.add_argument("--batch-size", type=int, default=2)
    command.add_argument("--judge-label", required=True, help="Actual model label selected in ChatGPT")
    command.add_argument("--ui-settings", default="fresh temporary chat; memory/custom instructions off; tools off")
    command.add_argument("--calibration-report", help="Required for main; freeze the same manual judge used in calibration")
    command = commands.add_parser("import", help="Validate a complete pass of saved JSON responses")
    command.add_argument("--bundle", required=True)
    command.add_argument("--responses", required=True)
    command.add_argument("--session-log", required=True)
    command.add_argument("--pass-number", type=int, choices=(1, 2), required=True)
    command.add_argument("--output", required=True)
    command = commands.add_parser("repeatability")
    command.add_argument("--first", required=True)
    command.add_argument("--second", required=True)
    command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "export":
        from .export import export_bundle
        export_bundle(args)
    elif args.command == "import":
        from .results import import_results
        import_results(args)
    else:
        from .results import repeatability
        repeatability(args)


if __name__ == "__main__":
    main()
