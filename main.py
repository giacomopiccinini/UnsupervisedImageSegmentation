if __name__ == "__main__": 

    import torch
    from chronopy.Clock import Clock
    from Code.Parser.parser import parse
    from Code.Modules.A_train import train

    clock = Clock()

    # Parse arguments
    clock.lap("Parser")
    args = parse()

    # Train the model
    clock.lap("Segmentation")
    model = train(args)

    # Save the model
    clock.lap("Saving")
    torch.save(model.state_dict(), args.model_path)

    clock.lap()
    clock.summary()