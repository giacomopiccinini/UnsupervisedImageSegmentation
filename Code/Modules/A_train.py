def train(args):

    """Train Segmentation Network"""

    import torch
    import numpy as np

    from tqdm import tqdm
    from torch.optim import SGD
    from torch.utils.data import DataLoader

    from Code.Network.segnet import SegNet
    from Code.Loss.loss import loss_function
    from Code.Classes.Data import Data

    # Initialise network
    model = SegNet(args)

    # Enable CUDA and set up GPU
    if torch.cuda.is_available():
        model.cuda()

    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    # Set optimizer
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # Load dataset and loader for PyTorch
    dataset = Data(args.path)
    loader = DataLoader(dataset)

    print("\nTraining the model...")
    # Star training
    for epoch in range(args.epochs):

        print("\nEpoch:", epoch + 1, "/", args.epochs)

        # Load batch and discard file names
        for batch, _ in tqdm(loader):

            batch = batch.to(device=device)

            # Forwarding
            optimizer.zero_grad()

            # Pass the batch of images through SegNet
            response_map = model(batch)

            # Apply argmax on every row, keep only index (= class), throw away value
            _, indexed_images = torch.max(response_map, 1)

            # Find the number of unique labels that identify segmented regions
            n_labels = len(np.unique(indexed_images.data.cpu().numpy()))

            # Compute the loss function given the response map
            loss = loss_function(response_map, args)

            # Back propagate
            loss.backward()

            # Take a step in the optimizer
            optimizer.step()

        # Print summary on epoch end
        print("\nResults: ", "Number of labels :", n_labels, " | Loss :", loss.item())

        # Check that we do not obtain less labels than needed
        if n_labels <= args.min_classes:

            print("Reached minimum number of labels, exiting.")

            # If reached minimum, stop training
            break

    return model
