from flask import Flask, request, jsonify
from sample import Sample
from seg_score import GRUScore
from gru_fcn import GRU_FCN, Vanilla_GRU, FCN_1D, Squeeze_Excite
from auto_encoder import BP_AE
from mlp import MLP
from result_fusion import FuzzyLayer, FusedFuzzyDeepNet
from config import DEBUG

app = Flask(__name__)


def validate(data):
    pass


@app.route("/api", methods=["POST"])
def api():
    t = request.get_json()

    sample = Sample(t["data"], t["time_interval"])
    seg_points = sample.calc_seg_points()  # 计算分割点
    prediction = sample.predict()

    return jsonify(
        {
            "time_series": sample.time_series,  # 插值后的数据
            "seg_points": seg_points,
            "fault_diagnosis": prediction,
        }
    )


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=DEBUG)
