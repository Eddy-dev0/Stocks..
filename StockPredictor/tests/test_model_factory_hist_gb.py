from stock_predictor.core.models import ModelFactory


def test_hist_gb_classifier_rewrites_dict_class_weight_to_balanced() -> None:
    factory = ModelFactory(
        "hist_gb",
        overrides={
            "class_weight": {-1.0: 1.2, 0.0: 0.8, 1.0: 1.1},
            "max_iter": 5,
        },
    )

    model = factory.create("classification", calibrate=False)
    estimator = model.named_steps["estimator"]

    assert getattr(estimator, "class_weight", None) == "balanced"
