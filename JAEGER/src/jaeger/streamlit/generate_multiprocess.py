import pandas as pd
import subprocess
import shlex
import multiprocessing as mp
import os

# ===== 配置区 =====
csv_file = "/mnt/disk1/xueying/jtvae_att/jtvae-trans/Jaejer/JAEGER/src/jaeger/streamlit/r3.csv"
cuda_list = [2, 3, 4, 5]

base_cmd = (
    "CUDA_VISIBLE_DEVICES={cuda} python jaeger_generate.py "
    "--run_streamlit False "
    "--assay_id TB_4859 "
    "--cmpd {cmpd} "
    "--smiles {smiles} "
    "--sel_direction 8 "
    "--sel_sampling_strategy Deterministic "
    "--sel_sampling_density MegaDense "
    "--sim_cutoff 0.2 "
    "--filter_mols False "
    "--opt_direction Increase "
    "--calcu_log"
)

# ===== worker 定义 =====
def worker(cuda_id, task_queue):
    while True:
        task = task_queue.get()
        if task is None:
            break  # 结束信号

        idx, cmpd, smiles, total = task

        cmd = base_cmd.format(
            cuda=cuda_id,
            cmpd=shlex.quote(cmpd),
            smiles=shlex.quote(smiles)
        )

        print(f"\n[GPU {cuda_id}] [{idx+1}/{total}] Running:")
        print(cmd, flush=True)

        ret = subprocess.run(cmd, shell=True)

        if ret.returncode != 0:
            print(f"❌ [GPU {cuda_id}] Failed at row {idx}: {cmpd}", flush=True)
            break

# ===== 主逻辑 =====
if __name__ == "__main__":
    df = pd.read_csv(csv_file)
    total = len(df)

    print(f"共需运行 {total} 条任务")
    print(f"使用 GPU: {cuda_list}")

    task_queue = mp.Queue()

    # 放任务
    for idx, row in df.iterrows():
        task_queue.put((
            idx,
            str(row["ID"]),
            str(row["smiles"]),
            total
        ))

    # 每个 GPU 启一个进程
    workers = []
    for cuda in cuda_list:
        p = mp.Process(target=worker, args=(cuda, task_queue))
        p.start()
        workers.append(p)

    # 结束信号
    for _ in cuda_list:
        task_queue.put(None)

    for p in workers:
        p.join()

    print("✅ All tasks finished.")