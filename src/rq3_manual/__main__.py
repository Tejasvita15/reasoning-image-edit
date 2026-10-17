"""Export images for ChatGPT, import manual judgments, and analyze RQ3."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("export", help="Make an offline ZIP with blinded images and prompts")
    command.add_argument("--generations", required=True)
    command.add_argument("--prepared-data", default=os.environ.get("RQ3_PREPARED_DATA"),
                         help="Verified prepared HF data directory; defaults to RQ3_PREPARED_DATA")
    command.add_argument("--output-dir", required=True)
    command.add_argument("--kind", choices=("calibration", "main"), default="calibration")
    command.add_argument("--limit", type=int, default=20)
    command.add_argument("--seeds", type=int, nargs="+", default=[42])
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
    command = commands.add_parser("analyze", help="Analyze main manual scores with measured GPU timings")
    command.add_argument("--bundle", required=True)
    command.add_argument("--scores", required=True)
    command.add_argument("--generations", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--similarity", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        from .export import export_bundle
        export_bundle(args)
    elif args.command == "import":
        from .results import import_results
        import_results(args)
    elif args.command == "repeatability":
        from .results import repeatability
        repeatability(args)
    else:
        from .analysis import analyze_manual
        analyze_manual(args)


if __name__ == "__main__":
    main()
