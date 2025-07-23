from torch import nn

class MLP(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.flatten = nn.Flatten() # flattens an image to row-vector [<->], so rows are samples -> shape[n, 784]
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inp, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def __setattr__(self, name, value):
        print(f"__setattr__ called: {name} = {value}")
        super().__setattr__(name, value)

    # ---

    def print_info_():
        # print(type(nn.Sequential()))
        # print(type(nn.Module()))
        # print(type(model))
        # print(model)
        # print(model.parameters())
        # for param in model.parameters():
        #     print(param) # tensors
        print(f"Model structure: {model}\n\n")
        for name, param in model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values[:2] : {param[:2]} \n")

    def predict_(model, ch, h, w, device=None):
        if device is None:
            device = next(model.parameters()).device
        with torch.no_grad(): # temporarily changing global state - torch.is_grad_enabled()
            X = torch.rand(2, ch, h, w, device=device) # (N, C, H, W)
            model.eval()
            logits = model(X)
            print(logits.shape)
            pred_probab = nn.Softmax(dim=1)(logits) # along row, columns are entries
            print(pred_probab)
            y_pred = pred_probab.argmax(dim=1) # along row
            # y_pred = [y1, y2]
            print(f"Predicted class: {y_pred}")
