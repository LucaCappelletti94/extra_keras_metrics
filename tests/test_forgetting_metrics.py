from extra_keras_metrics import metrics
import os

def test_forgetting_metrics():
    files = {
        f[:-3] for f in  os.listdir("extra_keras_metrics") if f.endswith(".py")
    } - {"__init__", "__version__", "metric", "parametrized_metric"}
    forgotten_metrics = files - set([
        m.__name__ for m in metrics
    ])
    for forgot in forgotten_metrics:
        print("You still need to add the metric {forgot}".format(forgot=forgot))
    assert not forgotten_metrics