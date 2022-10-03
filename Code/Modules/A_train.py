def train(image):

    """ Train Segmentation Network"""

    import torch
    from torch.optim import SGD
    from Code.Utilities.loading_utils import load
    from Code.Networks.Architectures.segnet import SegNet
    from Code.Networks.Losses.loss import loss_function
    from Code.Modules.B_plot import plot

    # Load training parameters
    training_parameters     = load("train.yaml")
    architecture_parameters = load("architecture.yaml")

    globals().update(training_parameters)
    globals().update(architecture_parameters)

    # Initialise network
    model = SegNet(image.data.size(1))

    # Enable CUDA
    if torch.cuda.is_available():
        model.cuda()
    model.train()

    # Set optimizer
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(epochs):

        # Forwarding
        optimizer.zero_grad()

        # Pass image through convolutional network
        response_map = model(image.data)[0]

        # Reshape image
        response_map = response_map.permute(1, 2, 0).contiguous().view(-1, n_classes)

        # Plot current status of segmentation and extract unique labels
        labels = plot(image, response_map)

        # Compute loss function
        loss = loss_function(response_map, image.shape[1], image.shape[0])

        # Back propagate
        loss.backward()

        # Take a step in the optimizer
        optimizer.step()

        # Print summary on epoch end
        print(epoch, '/', epochs, '|', ' label num :', labels, ' | loss :', loss.item())

        # Check that we do not obtain less labels than needed
        if labels <= min_classes:

            print("Reached minimum number of labels, exiting.")

            # If reached minimum, stop training
            break

    torch.save(model, "Model/model.pt")
