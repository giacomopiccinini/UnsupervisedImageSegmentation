if __name__ == "__main__":

    import torch
    from chronopy.Clock import Clock
    from Code.Parser.parser import parse
    from Code.Modules.A_train import train
    from Code.Modules.B_plot import plot

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

    # Plot the segmentation for each image in the train set
    clock.lap("Saving segmentation")
    plot(model, args)    

    clock.lap()
    clock.summary()
