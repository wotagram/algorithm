import torch


def train_model(model, train_loader, num_epochs=128):
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for it in range(num_epochs):
        losses = []
        for x, y in train_loader:
            if cuda:
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            try:
                outputs = model(x)
                loss = loss_fn(outputs.squeeze(), y.type(torch.float32))
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            except IndexError as e:
                print("IndexError occurred:")
                print(e)
                print("x shape:", x.shape)
                print("y shape:", y.shape)
                # Add more debug information as needed

        if len(losses) > 0:
            avg_loss = sum(losses) / len(losses)
            print("iter #{} Loss: {:.4f}".format(it, avg_loss))
        else:
            print("iter #{} No losses recorded.".format(it))

    return model
