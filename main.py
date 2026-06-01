import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py {train|sample} [args...]")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "train":
        from tabdlm.cli.train import main as train_main
        train_main()
    elif command == "sample":
        from tabdlm.cli.sample import main as sample_main
        sample_main()
    else:
        print(f"Unknown command: {command!r}. Use 'train' or 'sample'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
