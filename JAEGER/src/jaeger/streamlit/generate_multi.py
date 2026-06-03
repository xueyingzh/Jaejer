import pandas as pd
import subprocess
import shlex

# ===== 配置区 =====
csv_file = "/mnt/disk1/xueying/jtvae_att/jtvae-trans/Jaejer/JAEGER/src/jaeger/streamlit/r2.csv"
cuda_id = 5

base_cmd = (
    "CUDA_VISIBLE_DEVICES={cuda} python jaeger_generate.py "
    "--run_streamlit False "
    "--assay_id TB_4859 "
    "--cmpd {cmpd} "
    "--smiles {smiles} "
    "--sel_direction 8 "
    "--sel_sampling_strategy Deterministic "
    "--sel_sampling_density Dense "
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
    cmpd = str(row["ID"])
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

# mega_cmd = (
#     "CUDA_VISIBLE_DEVICES={cuda} python jaeger_generate.py "
#     "--run_streamlit False "
#     "--assay_id TB_4859 "
#     "--cmpd {cmpd} "
#     "--smiles {smiles} "
#     "--sel_direction 8 "
#     "--sel_sampling_strategy Deterministic "
#     "--sel_sampling_density MegaDense "
#     "--sim_cutoff 0.2 "
#     "--filter_mols False "
#     "--opt_direction Increase "
#     "--calcu_log"
# )

# # ===== 逐行运行 =====
# for idx, row in df.iterrows():
#     cmpd = str(row["ID"])
#     smiles = str(row["smiles"])

#     cmd = mega_cmd.format(
#         cuda=cuda_id,
#         cmpd=shlex.quote(cmpd),
#         smiles=shlex.quote(smiles)
#     )

#     print(f"\n[{idx+1}/{len(df)}] Running:")
#     print(cmd)

#     ret = subprocess.run(cmd, shell=True)

#     if ret.returncode != 0:
#         print(f"❌ Failed at row {idx}: {cmpd}")
#         break