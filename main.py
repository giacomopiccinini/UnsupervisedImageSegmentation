if __name__ == "__main__": 

    import sys
    from pathlib import Path

    from Code.Modules.A_train import train
    from Code.Classes.Image import Image

    # Check arguments
    if len(sys.argv) != 2:
        print("Please use 'python main.py <FILENAME>' or 'python main.py <DIRECTORY>'")
        exit(1)

    # Read path from command line
    path = Path(sys.argv[1]).absolute()

    # Check if argument is a file or a directory (and check if it exists)
    if path.is_dir():
        files = path.rglob("*") # If path is a directory, load all files recursively
    elif path.is_file():
        files = [path]          # If is a single file, load it alone
    else:
        raise Exception(f'ERROR: {path} does not exist') 

    filename = str(files[0])

    # Initialise image
    image = Image(filename)

    train(image)
