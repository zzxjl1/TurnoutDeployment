import torch
from torch import nn
from config import SUPPORTED_SAMPLE_TYPES, FORCE_CPU
from utils import load_model

FILE_PATH = "./models/result_fusion.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
# print("Using device:", DEVICE)
N_CLASSES = len(SUPPORTED_SAMPLE_TYPES)
INPUT_VECTOR_SIZE = 3 * N_CLASSES  # 输入向量大小


class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FuzzyLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        fuzzy_degree_weights = torch.Tensor(self.input_dim, self.output_dim)
        self.fuzzy_degree = nn.Parameter(fuzzy_degree_weights)
        sigma_weights = torch.Tensor(self.input_dim, self.output_dim)
        self.sigma = nn.Parameter(sigma_weights)

        # initialize fuzzy degree and sigma parameters
        nn.init.xavier_uniform_(self.fuzzy_degree)  # fuzzy degree init
        nn.init.ones_(self.sigma)  # sigma init

    def forward(self, input):
        fuzzy_out = []
        for variable in input:
            fuzzy_out_i = torch.exp(
                -torch.sum(
                    torch.sqrt((variable - self.fuzzy_degree) / (self.sigma**2))
                )
            )
            if torch.isnan(fuzzy_out_i):
                fuzzy_out.append(variable)
            else:
                fuzzy_out.append(fuzzy_out_i)
        return torch.tensor(fuzzy_out, dtype=torch.float)


class FusedFuzzyDeepNet(nn.Module):
    def __init__(
        self,
        input_vector_size,
        fuzz_vector_size,
        num_class,
        fuzzy_layer_input_dim=1,
        fuzzy_layer_output_dim=1,
        dropout_rate=0.2,
    ):
        super(FusedFuzzyDeepNet, self).__init__()
        self.input_vector_size = input_vector_size
        self.fuzz_vector_size = fuzz_vector_size
        self.num_class = num_class
        self.fuzzy_layer_input_dim = fuzzy_layer_input_dim
        self.fuzzy_layer_output_dim = fuzzy_layer_output_dim

        self.dropout_rate = dropout_rate

        self.bn = nn.BatchNorm1d(self.input_vector_size)
        self.fuzz_init_linear_layer = nn.Linear(
            self.input_vector_size, self.fuzz_vector_size
        )

        fuzzy_rule_layers = []
        for i in range(self.fuzz_vector_size):
            fuzzy_rule_layers.append(
                FuzzyLayer(fuzzy_layer_input_dim, fuzzy_layer_output_dim)
            )
        self.fuzzy_rule_layers = nn.ModuleList(fuzzy_rule_layers)

        self.dl_linear_1 = nn.Linear(self.input_vector_size, self.input_vector_size)
        self.dl_linear_2 = nn.Linear(self.input_vector_size, self.input_vector_size)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.fusion_layer = nn.Linear(
            self.input_vector_size * 2, self.input_vector_size
        )
        self.output_layer = nn.Linear(self.input_vector_size, self.num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        input = self.bn(input)
        fuzz_input = self.fuzz_init_linear_layer(input)
        fuzz_output = torch.zeros(input.size(), dtype=torch.float).to(DEVICE)
        for col_idx in range(fuzz_input.size()[1]):
            col_vector = fuzz_input[:, col_idx : col_idx + 1]
            fuzz_col_vector = (
                self.fuzzy_rule_layers[col_idx](col_vector).unsqueeze(0).view(-1, 1)
            )
            fuzz_output[:, col_idx : col_idx + 1] = fuzz_col_vector

        dl_layer_1_output = torch.sigmoid(self.dl_linear_1(input))
        dl_layer_2_output = torch.sigmoid(self.dl_linear_2(dl_layer_1_output))
        dl_layer_2_output = self.dropout_layer(dl_layer_2_output)

        cat_fuzz_dl_output = torch.cat([fuzz_output, dl_layer_2_output], dim=1)

        fused_output = torch.sigmoid(self.fusion_layer(cat_fuzz_dl_output))
        fused_output = torch.relu(fused_output)

        output = self.softmax(self.output_layer(fused_output))

        return output


model = FusedFuzzyDeepNet(
    input_vector_size=INPUT_VECTOR_SIZE,
    fuzz_vector_size=8,
    num_class=N_CLASSES
).to(DEVICE)  # FNN模型
model.load_state_dict(load_model(FILE_PATH, DEVICE))


def model_input_parse(bp_result, gru_fcn_result, ae_result, batch_simulation=True):
    # 拼接三个分类器的结果
    result = torch.cat([bp_result, gru_fcn_result, ae_result], dim=0)
    if batch_simulation:
        result = result.unsqueeze(0)
    return result


def predict(bp_result, gru_fcn_result, ae_result):
    model_input = model_input_parse(bp_result, gru_fcn_result, ae_result)
    model.eval()
    with torch.no_grad():
        output = model(model_input)
    return output.squeeze()
