if __name__ == "__main__": 

    import torch
    from Code.Parser.parser import parse
    from Code.Modules.A_train import train

    # Parse arguments
    args = parse()

    # Train the model
    model = train(args)

    # Save the model
    torch.save(model.state_dict(), args.model_path)