import torch
import torch.nn as nn

# code adapted from https://github.com/DeepMicroscopy/Cox_AMIL/blob/master/models/model_amil.py

class PatientAttnNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=False, n_classes=1):
        super(PatientAttnNet, self).__init__()
        self.attention_a = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()]

        self.attention_b = [
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class PatientAMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, gate=True, dropout=False, n_classes=1):
        super().__init__()
        fc = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = PatientAttnNet(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout, n_classes=1)
        else:
            attention_net = PatientAttnNet(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.predictor = nn.Linear(hidden_dim, n_classes)
        self.n_classes = n_classes

    def forward(self, h, return_features=False, attention_only=False):
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = nn.functional.softmax(A, dim=1)
        M = torch.mm(A, h)
        risk = self.predictor(M)

        if return_features:
            results_dict = {'features': M}
        else:
            results_dict = {}

        return risk, A_raw, results_dict