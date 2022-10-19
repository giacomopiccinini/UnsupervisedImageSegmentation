if __name__ == "__main__": 

    from Code.Parser.parser import parse
    from Code.Modules.A_train import train

    args = parse()

    train(args)
