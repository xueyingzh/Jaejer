import pandas as pd
import subprocess
import shlex

# ===== 配置区 =====
csv_file = "r3b1.csv"
cuda_id = 4

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

# ===== 读取 CSV =====
df = pd.read_csv(csv_file)

print(f"共需运行 {len(df)} 条任务")

# ===== 逐行运行 =====
for idx, row in df.iterrows():
    cmpd = str(row["top_hit"])
    smiles = str(row["smiles"])

    cmd = base_cmd.format(
        cuda=cuda_id,
        cmpd=shlex.quote(cmpd),
        smiles=shlex.quote(smiles)
    )

    print(f"\n[{idx+1}/{len(df)}] Running:")
    print(cmd)

    ret = subprocess.run(cmd, shell=True)

    if ret.returncode != 0:
        print(f"❌ Failed at row {idx}: {cmpd}")
        break
