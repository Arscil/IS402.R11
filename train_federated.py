"""
Federated training with TensorFlow Federated (FedAvg) on the 5 simulated clients.

Each round: the server sends the global model to every client, each client
trains locally on its own data, the server averages the updates weighted by the
number of samples. Only model weights travel; client data never leaves its .npz.

Usage (inside the `fl` environment, from the project root):
    python train_federated.py --scenario iid
    python train_federated.py --scenario noniid --rounds 30 --local-epochs 1
    python train_federated.py --scenario iid --local-epochs 3 --name fl_iid_e3
Outputs:
    models/<name>.keras                (default name: fl_<scenario>)
    results/history_<name>.json
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

import config
import data_utils
import model_utils


def main():
    parser = argparse.ArgumentParser(description="Federated training (FedAvg) with TFF")
    parser.add_argument("--scenario", choices=["iid", "noniid"], default="iid")
    parser.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=config.CLIENT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.CLIENT_BATCH_SIZE)
    parser.add_argument("--client-lr", type=float, default=config.CLIENT_LEARNING_RATE)
    parser.add_argument("--server-lr", type=float, default=config.SERVER_LEARNING_RATE)
    parser.add_argument("--class-weight", choices=["balanced", "sqrt", "none"],
                        default=config.CLASS_WEIGHT_MODE)
    parser.add_argument("--name", default=None, help="Experiment name (default: fl_<scenario>)")
    args = parser.parse_args()
    name = args.name or f"fl_{args.scenario}"

    # Client data stays per client; the test set is only used by the "server" to report progress.
    client_datasets = data_utils.make_client_datasets(
        scenario=args.scenario, batch_size=args.batch_size, epochs=args.local_epochs)
    client_stats = data_utils.get_client_stats(args.scenario)
    _, y_train, X_test, y_test = data_utils.load_data()

    # Simplification of the simulation (ghi vào báo cáo): class weights are computed on the
    # union of client data. In a real deployment each client would only know its own ratio.
    class_weights = data_utils.get_class_weights(y_train, args.class_weight)
    element_spec = client_datasets[0].element_spec
    print(f"Kịch bản: {args.scenario} | {len(client_datasets)} client | class weights: {class_weights}")
    for s in client_stats:
        print(f"  Client {s['client_id'] + 1}: {s['total_samples']:,} mẫu, {s['fraud_samples']} gian lận "
              f"({s['fraud_rate_pct']}%)")

    def model_fn():
        return tff.learning.models.from_keras_model(
            keras_model=model_utils.build_keras_model(),
            input_spec=element_spec,
            loss=model_utils.make_loss(class_weights),
            metrics=model_utils.make_keras_metrics(),
        )

    process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=args.client_lr),
        server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=args.server_lr),
    )

    eval_model = model_utils.build_keras_model()
    log = {"name": name, "type": "federated", "step_unit": "round", "scenario": args.scenario,
           "config": {"rounds": args.rounds, "local_epochs": args.local_epochs,
                      "batch_size": args.batch_size, "client_lr": args.client_lr,
                      "server_lr": args.server_lr, "num_clients": len(client_datasets),
                      "class_weight_mode": args.class_weight, "class_weights": class_weights,
                      "scaler": config.SCALER_TYPE},
           "clients": client_stats,
           "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
           "steps": []}

    t0 = time.perf_counter()
    state = process.initialize()
    for round_num in range(1, args.rounds + 1):
        output = process.next(state, client_datasets)  # LearningProcessOutput(state, metrics)
        state = output.state
        train = output.metrics["client_work"]["train"]

        process.get_model_weights(state).assign_weights_to(eval_model)
        prob = eval_model.predict(X_test, batch_size=4096, verbose=0)
        test = model_utils.compute_metrics(y_test, prob)

        entry = {"step": round_num, "elapsed_sec": round(time.perf_counter() - t0, 2),
                 **{f"train_{k}": float(np.asarray(v)) for k, v in train.items()},
                 **{f"test_{k}": v for k, v in test.items()}}
        log["steps"].append(entry)
        print(f"Round {round_num:>3}: train loss={entry['train_loss']:.4f} | test precision="
              f"{test['precision']:.4f} recall={test['recall']:.4f} f1={test['f1']:.4f} "
              f"pr_auc={test['pr_auc']:.4f} | {entry['elapsed_sec']}s")
    log["train_time_sec"] = round(time.perf_counter() - t0, 2)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_model.save(config.model_file(name))
    with open(config.history_file(name), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"Thời gian huấn luyện: {log['train_time_sec']} giây")
    print(f"Đã lưu mô hình: {config.model_file(name)}")
    print(f"Đã lưu lịch sử: {config.history_file(name)}")


if __name__ == "__main__":
    main()
